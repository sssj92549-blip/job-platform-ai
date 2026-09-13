"""本地中文向量化与Chroma检索；严格按Java提供的简历版本白名单查询。"""

import hashlib
from threading import RLock

import numpy as np

from .config import Settings
from .errors import ServiceError
from .schemas import SearchRequest, VectorRequest


class VectorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self.collection = self.tokenizer = None
        self.client = None
        self.lock = RLock()

    def close(self):
        """关闭Chroma文件句柄，支持Windows下释放索引目录。"""
        with self.lock:
            if self.client is not None:
                self.client.close()
                self.client = self.collection = None

    def warmup(self):
        with self.lock:
            if self.collection is not None:
                return
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                from fastembed import TextEmbedding
                from tokenizers import Tokenizer

                cfg = self.settings
                if self.model is None:
                    self.model = TextEmbedding(
                        model_name=cfg.embedding.model,
                        cache_dir=str(cfg.storage.model_cache),
                        threads=2,
                    )
                    self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
                    self.tokenizer.no_truncation()
                    self.tokenizer.no_padding()
                dimension = len(self.embed("初始化维度校验"))
                signature = hashlib.sha256(
                    f"{cfg.embedding.model}:{cfg.embedding.chunk_chars}:weighted-mean-v1:{dimension}".encode()
                ).hexdigest()
                client = chromadb.PersistentClient(
                    path=str(cfg.storage.chroma_path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self.client = client
                collection = client.get_or_create_collection(
                    cfg.embedding.collection,
                    embedding_function=None,
                    metadata={"signature": signature},
                    configuration={"hnsw": {"space": "cosine"}},
                )
                if collection.metadata.get("signature") != signature:
                    raise ValueError("embedding configuration mismatch")
                self.collection = collection
            except Exception:
                raise ServiceError(
                    503, 50301, "向量模型或Chroma不可用；更换模型须新建集合"
                ) from None

    def chunks(self, text: str) -> list[str]:
        """逐段保留全文，并按真实token数进一步切分，不依赖模型的静默截断。"""

        def split(part):
            if len(self.tokenizer.encode(part, add_special_tokens=False).ids) <= 400:
                return [part]
            middle = len(part) // 2
            if middle == 0:
                raise ServiceError(422, 42201, "文本无法分块")
            return split(part[:middle]) + split(part[middle:])

        size = self.settings.embedding.chunk_chars
        return [
            chunk
            for start in range(0, len(text), size)
            for chunk in split(text[start : start + size])
        ]

    def embed(self, text: str) -> list[float]:
        parts = self.chunks(text)
        rows = list(self.model.embed(parts, batch_size=16))
        pooled = np.average(np.asarray(rows), axis=0, weights=[len(p) for p in parts])
        norm = np.linalg.norm(pooled)
        if not np.isfinite(norm) or norm == 0:
            raise ServiceError(503, 50301, "向量编码结果无效")
        return (pooled / norm).tolist()

    def upsert(self, resume_id: str, version: int, request: VectorRequest) -> dict:
        with self.lock:
            self.warmup()
            try:
                vector = self.embed(request.text)
                self.collection.upsert(
                    ids=[f"{resume_id}:{version}"],
                    embeddings=[vector],
                    metadatas=[
                        {
                            "resumeId": resume_id,
                            "resumeVersion": version,
                            "candidateId": request.candidateId,
                            **request.metadata.model_dump(),
                        }
                    ],
                )
                return {
                    "resumeId": resume_id,
                    "resumeVersion": version,
                    "indexed": True,
                    "embeddingModel": self.settings.embedding.model,
                    "dimension": len(vector),
                }
            except ServiceError:
                raise
            except Exception:
                raise ServiceError(503, 50301, "向量写入失败") from None

    def delete(self, resume_id: str, version: int) -> dict:
        with self.lock:
            self.warmup()
            try:
                self.collection.delete(ids=[f"{resume_id}:{version}"])
            except Exception:
                raise ServiceError(503, 50301, "向量删除失败") from None
            return {"resumeId": resume_id, "resumeVersion": version, "deleted": True}

    def search(self, request: SearchRequest) -> dict:
        result = {"jobId": request.jobId, "jobVersion": request.jobVersion, "matches": []}
        if not request.eligibleResumes:
            return result
        with self.lock:
            self.warmup()
            try:
                vector = self.embed(request.queryText)
                allowed = sorted(
                    {f"{r.resumeId}:{r.resumeVersion}" for r in request.eligibleResumes}
                )
                matches = []
                # 分批过滤白名单后再取TopK，不能全库检索后过滤（会漏掉有效候选人）。
                for start in range(0, len(allowed), 500):
                    batch = allowed[start : start + 500]
                    rows = self.collection.query(
                        query_embeddings=[vector],
                        ids=batch,
                        where={"$and": [{"confirmed": True}, {"discoverable": True}]},
                        n_results=len(batch),
                        include=["metadatas", "distances"],
                    )
                    for identity, meta, distance in zip(
                        rows["ids"][0], rows["metadatas"][0], rows["distances"][0]
                    ):
                        if (
                            identity not in batch
                            or not meta["confirmed"]
                            or not meta["discoverable"]
                        ):
                            continue
                        similarity = float(np.clip(1 - distance, 0, 1))
                        if similarity >= request.minSimilarity:
                            matches.append(
                                {
                                    "resumeId": meta["resumeId"],
                                    "resumeVersion": meta["resumeVersion"],
                                    "candidateId": meta["candidateId"],
                                    "similarity": similarity,
                                }
                            )
                matches.sort(
                    key=lambda row: (-row["similarity"], int(row["resumeId"]), row["resumeVersion"])
                )
                result["matches"] = matches[: request.topK]
                return result
            except Exception:
                raise ServiceError(503, 50301, "人才向量检索失败") from None
