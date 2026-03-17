import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from download_model import download_if_needed
from app.models.model_loader import get_model
from app.routes.predict import router

load_dotenv()

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    download_if_needed()
    get_model()
    logger.info("Model ready. API is live.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Tomato Leaf Disease Diagnostic API",
    description = (
        "AI-powered API that classifies tomato leaf diseases from smartphone photos "
        "and generates diagnostic reports with treatment recommendations."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS - allow Vercel frontend + local dev
VERCEL_URL = os.getenv("VERCEL_URL", "")  

allowed_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

if VERCEL_URL:
    allowed_origins.append(VERCEL_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = False,
)

app.include_router(router)


@app.get("/ui", include_in_schema=False)
def serve_farmer_ui():
    return FileResponse("interface.html")


@app.get("/review", include_in_schema=False)
def serve_review_ui():
    return FileResponse("review.html")