from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from logging.config import dictConfig
from pathlib import Path
from api.routes.graphrag import vector_store, llm
from api.routes import graphrag, query

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "phase4.log"

dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(LOG_FILE),
                "mode": "w",
                "encoding": "utf-8",
                "formatter": "standard",
            }
        },
        "root": {"level": "DEBUG", "handlers": ["file"]},
        "loggers": {
            "uvicorn": {"handlers": ["file"], "level": "INFO", "propagate": False},
            "uvicorn.error": {
                "handlers": ["file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["file"],
                "level": "INFO",
                "propagate": False,
            },
            "openai": {"handlers": ["file"], "level": "DEBUG", "propagate": False},
            "httpx": {"handlers": ["file"], "level": "INFO", "propagate": False},
            "httpcore": {"handlers": ["file"], "level": "DEBUG", "propagate": False},
        },
    }
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_ok = await llm.check_connection()
    if llm_ok:
        logger.info("LLM connected, less go! 🚀")
    else:
        logger.warning("LLM connection failed :((")
    await vector_store.connect()
    await vector_store.create_index()
    yield
    await vector_store.close()


app = FastAPI(title="Phase - 4 - Knowledge Graph", lifespan=lifespan)

app.include_router(graphrag.router, prefix="/api", tags=["graphrag"])
app.include_router(query.router, prefix="/api", tags=["graphquery"])


@app.get("/")
def hello():
    return {"message": "Hello from Phase-4"}
