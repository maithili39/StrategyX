import os
import yaml
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt

# Workaround for passlib + bcrypt >= 4.0.0 issue where passlib expects bcrypt.__about__.__version__
# which was removed in recent bcrypt versions.
try:
    import bcrypt
    if not hasattr(bcrypt, "__about__"):
        class DummyAbout:
            __version__ = bcrypt.__version__
        bcrypt.__about__ = DummyAbout()
except ImportError:
    pass

from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Load security configs
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Default Admin Staff Credential
ADMIN_USERNAME = "strategyx_admin"
# Bcrypt hash of "strategyx_password"
ADMIN_PASSWORD_HASH = "$2b$12$np06.wiClBuKpMJfjAmhCO3fnTYRf/Ka4RJfaxUf5vTdN9pUV0uPy"

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            sec_cfg = cfg.get("security", {})
            SECRET_KEY = sec_cfg.get("secret_key", SECRET_KEY)
            ALGORITHM = sec_cfg.get("algorithm", ALGORITHM)
            ACCESS_TOKEN_EXPIRE_MINUTES = sec_cfg.get("access_token_expire_minutes", ACCESS_TOKEN_EXPIRE_MINUTES)
    except Exception:
        pass

# Environment overrides
SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", ADMIN_USERNAME)

# Cryptography context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 dependency schema
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a signed JSON Web Token (JWT).
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    FastAPI security dependency validating token signature and expiry.
    Returns the username of the authenticated client.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    return username
