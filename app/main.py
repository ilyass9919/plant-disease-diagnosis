import logging
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
    logger.info("Starting up - loading model...")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(router)


@app.get("/ui", include_in_schema=False)   
def serve_ui():
    return FileResponse("interface.html")