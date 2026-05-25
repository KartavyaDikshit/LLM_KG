from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Triple(BaseModel):
    subject: str = Field(description="The source entity")
    predicate: str = Field(description="The relationship between subject and object")
    obj: str = Field(description="The target entity")
    confidence: float = Field(description="Confidence score for this extraction", ge=0, le=1)

class AgentState(TypedDict):
    input_text: str
    domain: str
    planner_strategy: Optional[str]
    extracted_triples: List[Triple]
    validation_feedback: Optional[str]
    is_valid: bool
    iterations: int
