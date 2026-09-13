class ServiceError(Exception):
    """对外使用稳定业务码，不输出第三方响应、密钥或文件系统路径。"""

    def __init__(self, status: int, code: int, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)
