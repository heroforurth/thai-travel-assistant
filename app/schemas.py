"""Every model output that crosses a boundary is parsed into one of these.

The agent speaks a small JSON protocol (the proxy ignores `response_format`, so we
ask for JSON in the prompt and validate it here). An invalid step is never executed:
the agent shows the model the validation error and asks it to try again.
"""
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from app import config


class SearchAction(BaseModel):
    action: Literal["search"]
    query: str = Field(min_length=2, max_length=300)
    destination: str | None = None

    @field_validator("destination", mode="before")
    @classmethod
    def known_destination(cls, v):
        # Allow-list: the model may only filter on destinations we actually indexed.
        if v in (None, "", "null", "none", "None"):
            return None
        for d in config.DESTINATIONS:
            if str(v).strip().lower() == d.lower():
                return d
        raise ValueError(f"destination must be one of {config.DESTINATIONS} or null")


class CalculateAction(BaseModel):
    action: Literal["calculate"]
    expression: str = Field(min_length=1, max_length=200, pattern=r"^[0-9+\-*/().%\s]+$")


class FinalAction(BaseModel):
    action: Literal["final"]
    answer: str = Field(min_length=1, max_length=2000)
    citations: list[str] = Field(default_factory=list, max_length=10)


AgentAction = Annotated[SearchAction | CalculateAction | FinalAction, Field(discriminator="action")]
_ACTION = TypeAdapter(AgentAction)


def extract_json(text: str) -> dict:
    """Pull the first {...} object out of a reply that may include ``` fences or prose."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group())
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def parse_action(text: str) -> SearchAction | CalculateAction | FinalAction:
    """Raises ValueError (incl. pydantic.ValidationError) if the reply is not a valid step."""
    return _ACTION.validate_python(extract_json(text))


class Citation(BaseModel):
    id: str
    destination: str
    section: str
    snippet: str


class AgentResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    steps: int
    stopped_by: Literal["final", "max_steps", "parse_error"]
    parse_failures: int = 0
    dropped_citations: list[str] = Field(default_factory=list)  # cited ids never retrieved
    trace: list[dict] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
