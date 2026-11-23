"""Client activity events. The server resolves the course from the resource."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ResourceActivityRequest(BaseModel):
    eventId: UUID
    resourceType: Literal["material", "video"]
    resourceId: int = Field(gt=0)
    activeSeconds: int = Field(ge=0, le=30)
