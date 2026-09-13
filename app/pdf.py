"""普通PDF直接提取文本；扫描页或混合页面降级PaddleOCR。"""

from pathlib import Path, PureWindowsPath
from threading import Lock

import numpy as np
import pymupdf

from .config import Settings
from .errors import ServiceError


class PdfParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = None
        self.lock = Lock()

    def warmup(self):
        with self.lock:
            if self.engine is None:
                try:
                    from paddleocr import PaddleOCR

                    self.engine = PaddleOCR(
                        lang=self.settings.ocr.language,
                        text_detection_model_name=self.settings.ocr.detection_model,
                        text_recognition_model_name=self.settings.ocr.recognition_model,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        device="cpu",
                        enable_mkldnn=False,
                    )
                except Exception:
                    raise ServiceError(503, 50301, "OCR模型未就绪，请检查安装与模型缓存") from None

    def read_ocr(self, image: np.ndarray) -> str:
        self.warmup()
        with self.lock:
            try:
                results = self.engine.predict(image)
                return "\n".join(str(text) for result in results for text in result["rec_texts"])
            except Exception:
                raise ServiceError(422, 42201, "扫描页识别失败") from None

    def safe_file(self, relative: str) -> Path:
        path = Path(relative.replace("\\", "/"))
        if (
            path.is_absolute()
            or PureWindowsPath(relative).drive
            or ".." in path.parts
            or ":" in relative
        ):
            raise ServiceError(400, 40001, "仅允许uploads下的相对PDF路径")
        root = self.settings.storage.uploads_root.resolve()
        target = root / path
        if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
            raise ServiceError(400, 40001, "不允许符号链接文件")
        resolved = target.resolve()
        if not resolved.is_relative_to(root):
            raise ServiceError(400, 40001, "文件路径越界")
        if resolved.suffix.lower() != ".pdf":
            raise ServiceError(400, 40001, "只支持PDF简历")
        if not resolved.is_file():
            raise ServiceError(404, 40401, "简历文件不存在")
        return resolved

    def extract(self, relative: str) -> dict:
        path = self.safe_file(relative)
        try:
            with path.open("rb") as file:
                content = file.read(10 * 1024 * 1024 + 1)
        except OSError:
            raise ServiceError(404, 40401, "简历文件不可读取") from None
        if len(content) > 10 * 1024 * 1024:
            raise ServiceError(413, 41301, "PDF不能超过10MB")
        if not content.startswith(b"%PDF-"):
            raise ServiceError(400, 40001, "文件不是有效PDF")
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception:
            raise ServiceError(422, 42201, "PDF损坏或无法读取") from None
        with doc:
            if doc.needs_pass:
                raise ServiceError(400, 40001, "不支持加密PDF")
            if not 1 <= len(doc) <= 20:
                raise ServiceError(400, 40001, "PDF页数须为1至20页")
            pages, methods = [], set()
            for page in doc:
                text = page.get_text("text", sort=True).strip()
                # 含较大图片的混合页也做整页OCR，避免只取页眉却丢掉图片中的履历。
                blocks = page.get_image_info()
                large_image = any(
                    pymupdf.Rect(b["bbox"]).get_area() > page.rect.get_area() * 0.15 for b in blocks
                )
                if len("".join(text.split())) < self.settings.ocr.text_threshold or large_image:
                    scale = self.settings.ocr.dpi / 72
                    if page.rect.width * page.rect.height * scale * scale > 16000000:
                        raise ServiceError(400, 40001, "PDF页面尺寸过大")
                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False
                    )
                    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, 3
                    )
                    text = self.read_ocr(image[:, :, ::-1].copy())
                    methods.add("OCR")
                else:
                    methods.add("TEXT")
                pages.append(text)
                if sum(map(len, pages)) + len(pages) - 1 > 60000:
                    raise ServiceError(422, 42201, "简历全文超过60000字，请精简")
            full_text = "\n".join(pages).strip()
            if not full_text:
                raise ServiceError(422, 42201, "无法识别简历文本")
            return {
                "extractedText": full_text,
                "extractionMethod": "MIXED" if len(methods) > 1 else next(iter(methods)),
                "pageCount": len(doc),
            }
