"""HTTP层仅处理认证、参数、信封；PDF/模型/检索逻辑在独立服务中。"""

import logging
import re
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from . import schemas as s
from .config import Settings, load_settings
from .errors import ServiceError
from .llm import DeepSeekClient
from .pdf import PdfParser
from .service import AiService
from .vector import VectorService

logger = logging.getLogger("job_platform_ai")


def create_app(settings: Settings | None = None, *, llm=None, pdf=None, vector=None, warmup=True):
    cfg = settings or load_settings()
    llm, pdf, vector = (
        llm or DeepSeekClient(cfg),
        pdf or PdfParser(cfg),
        vector or VectorService(cfg),
    )

    service = AiService(llm, pdf, vector)

    @asynccontextmanager
    async def lifespan(app):
        if not cfg.server.internal_token.get_secret_value():
            raise RuntimeError("请先在local.yml配置server.internal_token")
        if warmup:
            for service in (pdf, vector):
                try:
                    await run_in_threadpool(service.warmup)
                except ServiceError as error:
                    logger.warning("模型预热失败: %s", error.message)
        yield
        llm.close()
        vector.close()

    app = FastAPI(
        title="Job Platform AI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def envelope(request, data=None, *, code=0, message="success", status=200):
        return JSONResponse(
            status_code=status,
            content={
                "code": code,
                "message": message,
                "data": data,
                "requestId": request.state.request_id,
            },
        )

    @app.middleware("http")
    async def internal_security(request: Request, call_next):
        trace = request.headers.get("X-Request-Id", "")
        request.state.request_id = (
            trace if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace) else str(uuid.uuid4())
        )
        expected = cfg.server.internal_token.get_secret_value()
        supplied = request.headers.get("X-Internal-Token", "")
        if not expected or not secrets.compare_digest(expected.encode(), supplied.encode()):
            response = envelope(request, code=40101, message="内部认证失败", status=401)
        else:
            # 限制真实读取量，不能只相信Content-Length。
            size, chunks = 0, []
            async for chunk in request.stream():
                size += len(chunk)
                if size > 2 * 1024 * 1024:
                    response = envelope(request, code=41301, message="请求体过大", status=413)
                    break
                chunks.append(chunk)
            else:
                request._body = b"".join(chunks)
                response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return envelope(request, code=error.code, message=error.message, status=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return envelope(request, code=40001, message="请求参数不符合接口约定", status=400)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return envelope(
            request,
            code=error.status_code * 100 + 1,
            message="接口不存在或请求方法不正确",
            status=error.status_code,
        )

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        logger.error(
            "未处理异常 requestId=%s type=%s", request.state.request_id, type(error).__name__
        )
        return envelope(request, code=50001, message="AI服务内部错误", status=500)

    @app.get("/internal/health")
    def health(request: Request):
        ready = {
            "ocrReady": pdf.engine is not None,
            "vectorReady": vector.collection is not None,
            "llmConfigured": bool(cfg.deepseek.api_key.get_secret_value()),
        }
        if not all(ready.values()):
            missing = ",".join(key for key, value in ready.items() if not value)
            raise ServiceError(503, 50301, "依赖未就绪: " + missing)
        return envelope(request, {"status": "UP", **ready})

    @app.post("/internal/resumes/parse")
    def parse(body: s.ParseRequest, request: Request):
        return envelope(request, service.parse_resume(body))

    @app.post("/internal/ai/match")
    def match(body: s.MatchRequest, request: Request):
        return envelope(request, service.match(body))

    @app.post("/internal/ai/interview-questions")
    def interview(body: s.InterviewRequest, request: Request):
        return envelope(request, service.interview_questions(body))

    @app.post("/internal/ai/assistant")
    def assistant(body: s.AssistantRequest, request: Request):
        return envelope(request, service.assistant(body))

    @app.put("/internal/vector/resumes/{resumeId}/versions/{resumeVersion}")
    def upsert(
        body: s.VectorRequest,
        request: Request,
        resumeId: str = Path(pattern=r"^[1-9][0-9]{0,18}$"),
        resumeVersion: int = Path(ge=1),
    ):
        return envelope(request, service.upsert_resume(resumeId, resumeVersion, body))

    @app.delete("/internal/vector/resumes/{resumeId}/versions/{resumeVersion}")
    def delete(
        request: Request,
        resumeId: str = Path(pattern=r"^[1-9][0-9]{0,18}$"),
        resumeVersion: int = Path(ge=1),
    ):
        return envelope(request, service.delete_resume(resumeId, resumeVersion))

    @app.post("/internal/vector/talents/search")
    def search(body: s.SearchRequest, request: Request):
        return envelope(request, service.search_talents(body))

    return app


app = create_app()
