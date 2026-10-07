"""
auth.py — JWT authentication for Neural-Link.

Changes:
• Uses PyJWT ≥ 2.x API: jwt.encode() returns str (not bytes).
• Expiry stored as int (not float) to comply with RFC 7519.
• Fixed Researcher/Student role logic — passwords are now real strings (not None).
• Added get_current_user() dependency for protecting API endpoints.
• Python 3.10+ type hints.
"""

import time
import jwt
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from typing import Optional
import os

router     = APIRouter()
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "NEURAL_LINK_SUPER_SECRET_KEY_CHANGE_IN_PROD")
ALGORITHM  = "HS256"
TOKEN_TTL  = 3600   # seconds

# HTTPBearer scheme — reads the Authorization: Bearer <token> header
_bearer = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


def create_access_token(data: dict) -> str:
    payload = {**data, "exp": int(time.time()) + TOKEN_TTL, "iat": int(time.time())}
    # PyJWT ≥ 2.0 returns a str directly
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── Credential store ──────────────────────────────────────────────────────────
# Passwords are read from environment variables so they can be changed without
# touching source code.  Hard-coded defaults are intentionally weak and must be
# overridden in production via env vars.
_CREDENTIALS = {
    "admin":      (os.environ.get("ADMIN_PASSWORD",      "admin"),      "Admin"),
    "researcher": (os.environ.get("RESEARCHER_PASSWORD", "research123"), "Researcher"),
    "student":    (os.environ.get("STUDENT_PASSWORD",    "student123"),  "Student"),
}


@router.post("/token")
async def login(req: LoginRequest):
    entry = _CREDENTIALS.get(req.username)
    if entry is None or req.password != entry[0]:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    role = entry[1]
    token = create_access_token({"sub": req.username, "role": role})
    return {"access_token": token, "token_type": "bearer", "role": role}


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=403, detail=f"Could not validate credentials: {exc}")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
):
    """
    FastAPI dependency — validates the Bearer token on protected routes.

    Usage:
        @app.get("/protected")
        async def protected(user: dict = Depends(get_current_user)):
            ...

    The WebSocket /ws and public read-only endpoints (/health, /datasets,
    /model-stats, /model-comparison, /metrics) are intentionally left open
    so the frontend can connect without authenticating first.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return verify_token(credentials.credentials)
