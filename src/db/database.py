import os
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load DB URL from config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")

db_url = "sqlite:///strategyx.db"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            db_url = cfg.get("database", {}).get("url", db_url)
    except Exception:
        pass

# Environment variable override (takes precedence)
db_url = os.getenv("DATABASE_URL", db_url)

# SQLite specific config (needs check_same_thread=False for FastAPI/Streamlit multi-threading)
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(db_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Context manager dependency for yielding database sessions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
