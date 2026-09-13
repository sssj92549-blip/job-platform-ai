"""运行前先配置local.yml，默认仅监听本机8000端口。"""

import uvicorn
from app.config import load_settings

if __name__ == "__main__":
    settings = load_settings()
    # 单进程本地Chroma；避免多个worker重复加载OCR模型和争用索引。
    uvicorn.run("app.main:app", host=settings.server.host, port=settings.server.port, workers=1)
