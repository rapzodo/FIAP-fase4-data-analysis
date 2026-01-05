from pydantic import BaseModel, Field


class ExecutionError(BaseModel):
    error: str = Field(..., description="Error message")

