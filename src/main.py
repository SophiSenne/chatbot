import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada no arquivo .env")

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
    
    # 1. Carregar Modelo de Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Carregar o Vector Store Local (ChromaDB)
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    
    # DEBUG: Verificar se há documentos no banco
    collection = vectorstore._collection
    count = collection.count()
    print(f"✅ ChromaDB carregado: {count} documentos encontrados")
    
    if count == 0:
        print("⚠️ ATENÇÃO: Nenhum documento encontrado! Execute index_data.py primeiro.")
    
    # Configurar retriever com mais documentos
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Buscar top 4 chunks mais relevantes
    )
    
    # 3. Configurar o LLM (Gemini via API)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)
    
    # 4. Prompt para o LLM (melhorado)
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
    rag_chain, retriever = setup_rag_pipeline()

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
        
        # Retrieval: buscar documentos relevantes (SÍNCRONO)
        docs = retriever.invoke(pergunta.query)
        print(f"📚 Documentos recuperados: {len(docs)}")
        
        # Generation: gerar resposta com o LLM (SÍNCRONO)
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
        "rag_initialized": rag_chain is not None
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