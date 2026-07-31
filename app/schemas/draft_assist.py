from enum import Enum

from pydantic import BaseModel


class DraftAssistAction(str, Enum):
    CLARIFY = "CLARIFY"
    VERIFIABLE_REWRITE = "VERIFIABLE_REWRITE"
    SUGGEST_VARIABLES = "SUGGEST_VARIABLES"
    EXAMPLE = "EXAMPLE"


class DraftAssistRequest(BaseModel):
    action: DraftAssistAction
    text: str = ""
    category: str | None = None


class DraftAssistResponse(BaseModel):
    result: str
