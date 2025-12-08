import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agents.chatbot import router as chatbot_router, setup_rag_pipeline
from agents.fraud_detector import router as fraud_router, setup_fraud_pipeline
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada no arquivo .env")

app = FastAPI(
    title="RAG Compliance API",
    description="API de Retrieval-Augmented Generation para Compliance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)
app.include_router(fraud_router)

@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar o servidor para carregar o RAG pipeline."""
    try:
        setup_rag_pipeline()
    except Exception as e:
        print(f"ERRO na inicialização: {e}")
        raise

    try:
        setup_fraud_pipeline()
    except Exception as e:
        print(f"Aviso: pipeline de fraude não inicializado: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)