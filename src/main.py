import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Importação atualizada
from langchain_chroma import Chroma  # ✅ Importação atualizada
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada no arquivo .env")

# ========== CONFIGURAÇÃO DE CAMINHOS ==========
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

print(f"📂 Diretório base: {BASE_DIR}")
print(f"📂 Caminho do ChromaDB: {CHROMA_PATH}")
# =============================================

# --- Modelos Pydantic para a API ---
class Pergunta(BaseModel):
    query: str

class RespostaRAG(BaseModel):
    answer: str
    source_documents: list[str] = []

app = FastAPI()

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais para o pipeline RAG
rag_chain = None
retriever = None

def format_docs(docs):
    """Formata documentos para o contexto."""
    print(f"📄 Formatando {len(docs)} documentos recuperados")
    for i, doc in enumerate(docs):
        print(f"  Doc {i+1}: {doc.page_content[:100]}...")
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_pipeline():
    """Configura e retorna o pipeline RAG completo."""
    print("🚀 Iniciando a configuração do pipeline RAG...")
    
    # Verificar se o diretório existe
    if not CHROMA_PATH.exists():
        raise ValueError(
            f"❌ Diretório ChromaDB não encontrado em: {CHROMA_PATH}\n"
            f"Execute 'python index_data.py' primeiro."
        )
    
    # 1. Carregar Modelo de Embedding (MESMO modelo usado no index_data.py)
    print("🔧 Carregando modelo de embedding...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # ✅ Modelo completo
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Carregar o Vector Store Local (ChromaDB) com COLLECTION_NAME
    print(f"📦 Carregando ChromaDB de: {CHROMA_PATH}")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
        collection_name="compliance_docs"  # ✅ IMPORTANTE: mesmo nome usado no index_data.py
    )
    
    # DEBUG: Verificar se há documentos no banco
    collection = vectorstore._collection
    count = collection.count()
    print(f"✅ ChromaDB carregado: {count} documentos encontrados")
    
    if count == 0:
        raise ValueError(
            f"❌ Nenhum documento encontrado no ChromaDB!\n"
            f"Caminho verificado: {CHROMA_PATH}\n"
            f"Execute 'python index_data.py' para indexar os documentos."
        )
    
    # Configurar retriever com mais documentos
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # 3. Configurar o LLM (Gemini via API)
    print("🤖 Configurando Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)
    
    # 4. Prompt para o LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente de compliance da Dunder Mifflin Paper Company.

Use APENAS o contexto fornecido abaixo para responder à pergunta.
Se a resposta não estiver no contexto, diga: "Não encontrei essa informação no manual de compliance."

CONTEXTO:
{context}"""),
        ("human", "{question}")
    ])
    
    # 5. Criar a chain usando LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Pipeline RAG configurado com sucesso.")
    return rag_chain, retriever

@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar o servidor para carregar o RAG pipeline."""
    global rag_chain, retriever
    try:
        rag_chain, retriever = setup_rag_pipeline()
    except Exception as e:
        print(f"❌ ERRO na inicialização: {e}")
        raise

@app.get("/")
def read_root():
    return {"status": "RAG API running. Use /chat endpoint."}

@app.post("/chat", response_model=RespostaRAG)
async def chat_endpoint(pergunta: Pergunta):
    """Recebe uma pergunta, executa o RAG e retorna a resposta."""
    global rag_chain, retriever
    
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline não inicializado.")
    
    try:
        print(f"\n🔍 Nova pergunta: {pergunta.query}")
        
        # Retrieval: buscar documentos relevantes
        docs = retriever.invoke(pergunta.query)
        print(f"📚 Documentos recuperados: {len(docs)}")
        
        # Generation: gerar resposta com o LLM
        print("🤖 Gerando resposta com Gemini...")
        answer = rag_chain.invoke(pergunta.query)
        
        # Extrair textos dos documentos fonte
        source_texts = [doc.page_content[:200] + "..." for doc in docs]
        
        print(f"✅ Resposta gerada: {answer[:100]}...")
        
        return RespostaRAG(
            answer=answer,
            source_documents=source_texts
        )
        
    except Exception as e:
        print(f"❌ Erro ao processar a consulta: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar a pergunta: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Endpoint para verificar se o servidor está funcionando."""
    return {
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "chroma_path": str(CHROMA_PATH),
        "chroma_exists": CHROMA_PATH.exists()
    }

@app.get("/debug/test-retrieval")
def test_retrieval():
    """Endpoint para testar a recuperação de documentos."""
    global retriever
    
    if not retriever:
        return {"error": "Retriever não inicializado"}
    
    test_query = "relacionamentos românticos"
    docs = retriever.invoke(test_query)
    
    return {
        "query": test_query,
        "num_docs": len(docs),
        "docs": [
            {
                "content": doc.page_content[:300],
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    }