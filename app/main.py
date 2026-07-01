from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.vector_store.qdrant_client import create_collection_if_not_exists
from app.api.routes.ingest import router as ingest_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    #Startup 
    print(f"Smart AI Agent starting up.")
    print(f"  Chat model  : {settings.MISTRAL_CHAT_MODEL}")
    print(f"  Embed model : {settings.MISTRAL_EMBED_MODEL}")
    print(f"  Collection  : {settings.QDRANT_COLLECTION_NAME}")

   
    create_collection_if_not_exists()
    print("  Qdrant      : ready")

    yield 

   
    print("Smart AI Agent shutting down.")


app = FastAPI(
    title="Smart AI Agent",
    description="AI agent that reads documents, remembers context, and calls tools.",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)

@app.get("/health", tags=["System"])
async def health_check():
    """Quick liveness check — also verifies Qdrant connectivity."""
    from app.vector_store.qdrant_client import ping_qdrant
    qdrant_status = ping_qdrant()
    return {
        "status": "ok",
        "models": {
            "chat": settings.MISTRAL_CHAT_MODEL,
            "embed": settings.MISTRAL_EMBED_MODEL,
        },
        "qdrant": qdrant_status,
    }