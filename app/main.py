from fastapi import FastAPI

app = FastAPI(
    title="AI NewsOps Platform API",
    description="API for news classification, summarization, semantic search and MLOps monitoring.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-newsops-api",
        "version": "0.1.0",
    }


@app.get("/")
def root():
    return {
        "message": "AI NewsOps Platform API is running.",
        "docs": "/docs",
    }