class LightRAGError(Exception):
    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(LightRAGError):
    pass


class LLMError(LightRAGError):
    pass


class LLMUnavailableError(LLMError):
    pass


class IngestionError(LightRAGError):
    pass


class UnsupportedFormatError(IngestionError):
    pass


class NotFoundError(LightRAGError):
    pass


class DocumentNotFoundError(NotFoundError):
    pass


class SessionNotFoundError(NotFoundError):
    pass


class RetrievalError(LightRAGError):
    pass
