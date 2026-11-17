
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    username: str
    created_at: datetime


class UserRegistrationResponse(UserResponse):
    recovery_code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PasswordResetRequest(BaseModel):
    email: EmailStr
    recovery_code: str = Field(min_length=32, max_length=32)
    new_password: str = Field(min_length=8, max_length=128)