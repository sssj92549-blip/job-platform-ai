import json
from types import SimpleNamespace

import httpx
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import schemas as s
from app.config import load_settings
from app.errors import ServiceError
from app.llm import DeepSeekClient
from app.main import create_app
from app.pdf import PdfParser
from app.vector import VectorService


@pytest.fixture
def cfg(tmp_path):
    settings = load_settings()
    settings.server.internal_token = SecretStr("test-internal-token")
    settings.deepseek.api_key = SecretStr("unit-test-not-a-real-key")
    settings.storage.uploads_root = tmp_path
    settings.storage.chroma_path = tmp_path / "chroma"
    return settings


class FakeLlm:
    def generate(self, instruction, data, schema):
        values = {
            s.Parsed: dict(
                parsedName="张三",
                parsedPhone=None,
                parsedEducation="BACHELOR",
                parsedSkills=["Java"],
                parsedSummary=None,
            ),
            s.MatchResult: dict(score=85, reasons=["具备Java经验"], gaps=[]),
            s.AssistantResult: dict(answer="可以考虑Java开发岗位"),
            s.InterviewResult: dict(
                questions=[
                    dict(
                        number=i,
                        question="如何设计接口？",
                        direction="开发",
                        assessmentPoints=["接口设计"],
                    )
                    for i in range(1, 11)
                ]
            ),
        }
        return schema.model_validate(values[schema]).model_dump()

    def close(self):
        pass


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg, llm=FakeLlm(), warmup=False)) as client:
        client.headers["X-Internal-Token"] = "test-internal-token"
        yield client


JOB = dict(title="Java开发", description="开发接口", requirements="熟悉Java", skills=["Java"])


def test_auth_and_request_id(client):
    assert client.get("/internal/health", headers={"X-Internal-Token": "wrong"}).status_code == 401
    response = client.post(
        "/internal/ai/match",
        headers={"X-Request-Id": "java-trace-123"},
        json=dict(resumeText="Java项目", job=JOB),
    )
    assert response.json()["requestId"] == response.headers["X-Request-Id"] == "java-trace-123"
    assert response.json()["data"]["score"] == 85


def test_interview_and_assistant(client):
    response = client.post(
        "/internal/ai/interview-questions", json=dict(resumeText="Java", job=JOB, count=10)
    )
    assert len(response.json()["data"]["questions"]) == 10
    assert (
        client.post(
            "/internal/ai/interview-questions", json=dict(resumeText="Java", job=JOB, count=9)
        ).status_code
        == 400
    )
    assert client.post(
        "/internal/ai/assistant",
        json=dict(
            question="适合什么岗位",
            profile=dict(name="张三", education="BACHELOR", skills=["Java"]),
        ),
    ).json()["data"]["answer"]


def test_validation_and_health(client):
    assert client.post("/internal/ai/match", json={}).json()["code"] == 40001
    assert client.get("/internal/missing").status_code == 404
    response = client.get("/internal/health")
    assert response.status_code == 503 and response.json()["data"] is None
    assert (
        client.post("/internal/ai/match", content=b"x" * (2 * 1024 * 1024 + 1)).status_code == 413
    )


def test_pdf_text_parse(client, cfg):
    path = cfg.storage.uploads_root / "resume.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((60, 60), "Name: Zhang San; Skills: Java, MySQL; Education: Bachelor")
        doc.save(path)
    response = client.post(
        "/internal/resumes/parse", json=dict(resumeId="1", resumeVersion=1, filePath="resume.pdf")
    )
    assert response.status_code == 200
    assert response.json()["data"]["extractionMethod"] == "TEXT"
    assert response.json()["data"]["parsedName"] == "张三"


@pytest.mark.parametrize(
    "path", ["../secret.pdf", "C:/secret.pdf", "..\\secret.pdf", "/secret.pdf"]
)
def test_path_escape(client, path):
    response = client.post(
        "/internal/resumes/parse", json=dict(resumeId="1", resumeVersion=1, filePath=path)
    )
    assert response.status_code == 400


def test_scan_fallback_and_encrypted_pdf(cfg, monkeypatch):
    parser = PdfParser(cfg)
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(cfg.storage.uploads_root / "scan.pdf")
        doc.save(
            cfg.storage.uploads_root / "locked.pdf",
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="password",
        )
    monkeypatch.setattr(parser, "read_ocr", lambda image: "张三 Java 本科")
    assert parser.extract("scan.pdf")["extractionMethod"] == "OCR"
    with pytest.raises(ServiceError, match="加密"):
        parser.extract("locked.pdf")


def test_model_output_validation_and_timeout(cfg):
    llm = DeepSeekClient(cfg)
    llm.client.close()
    llm.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(dict(score=101, reasons=["x"], gaps=[]))
                            },
                        }
                    ]
                },
            )
        )
    )
    with pytest.raises(ServiceError) as error:
        llm.generate("score", {}, s.MatchResult)
    assert error.value.code == 42201
    llm.close()

    def timeout(request):
        raise httpx.ReadTimeout("timeout")

    llm.client = httpx.Client(transport=httpx.MockTransport(timeout))
    with pytest.raises(ServiceError) as error:
        llm.generate("score", {}, s.MatchResult)
    assert error.value.code == 50401
    llm.close()


def test_chroma_whitelist_metadata_and_idempotence(cfg, monkeypatch):
    import chromadb

    vector = VectorService(cfg)
    vector.collection = chromadb.PersistentClient(
        path=str(cfg.storage.chroma_path)
    ).create_collection(
        "unit_test", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    monkeypatch.setattr(vector, "embed", lambda text: [1.0, 0.0, 0.0])
    for rid, confirmed, visible in [
        ("1", True, True),
        ("2", True, False),
        ("3", False, True),
        ("4", True, True),
    ]:
        body = s.VectorRequest(
            candidateId=rid,
            text="Java",
            metadata=s.Metadata(confirmed=confirmed, discoverable=visible),
        )
        vector.upsert(rid, 1, body)
        vector.upsert(rid, 1, body)
    assert vector.collection.count() == 4
    query = s.SearchRequest(
        jobId="10",
        jobVersion=1,
        queryText="Java",
        topK=10,
        minSimilarity=0,
        eligibleResumes=[s.ResumeRef(resumeId=str(i), resumeVersion=1) for i in (1, 2, 3)],
    )
    assert [m["resumeId"] for m in vector.search(query)["matches"]] == ["1"]
    query.eligibleResumes = []
    assert vector.search(query)["matches"] == []
    vector.delete("1", 1)
    vector.delete("1", 1)
    assert vector.collection.count() == 3


def test_chunking_preserves_tail_and_pooling(cfg):
    vector = VectorService(cfg)
    vector.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: SimpleNamespace(ids=list(text))
    )
    vector.model = SimpleNamespace(
        embed=lambda parts, **kwargs: [np.array([len(p), 1.0]) for p in parts]
    )
    text = "Java招聘" * 1000 + "结尾不能丢弃"
    assert "".join(vector.chunks(text)) == text
    assert np.isclose(np.linalg.norm(vector.embed(text)), 1)
