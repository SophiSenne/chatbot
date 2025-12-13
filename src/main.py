import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agents.chatbot import router as chatbot_router, setup_rag_pipeline
from agents.fraud_detector import router as fraud_router, setup_fraud_pipeline
from agents.audit_agent import router as audit_router
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada no arquivo .env")

app = FastAPI(
    title="RAG Compliance & Audit API",
    description="API de Retrieval-Augmented Generation para Compliance e Investigação de Auditoria",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)
app.include_router(audit_router)
app.include_router(fraud_router)

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "RAG Compliance & Audit API",
        "version": "2.0.0",
        "endpoints": {
            "chatbot": "/chat",
            "audit_investigation": "/audit/investigate",
            "audit_file": "/audit/investigate-file",
            "audit_health": "/audit/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar o servidor para carregar o RAG pipeline."""
    try:
        print("🚀 Inicializando serviços...")
        setup_rag_pipeline()
        print("✅ RAG Pipeline configurado")
        print("✅ Audit Agent configurado")
        print("🎉 Servidor pronto!")
    except Exception as e:
        print(f"❌ ERRO na inicialização: {e}")
        raise

    try:
        setup_fraud_pipeline()
    except Exception as e:
        print(f"Aviso: pipeline de fraude não inicializado: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)