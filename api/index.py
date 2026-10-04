import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

BACKEND_DIR = (
    BASE_DIR /
    "backend"
)

sys.path.insert(
    0,
    str(BACKEND_DIR)
)


from app import app


# Vercel uses this Flask application