# auth/schemas.py
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "teacher"

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str