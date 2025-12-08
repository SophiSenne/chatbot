from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "../chroma_db"

print(f"Diretório base: {BASE_DIR}")
print(f"Caminho do ChromaDB: {CHROMA_PATH}")

class Pergunta(BaseModel):
    query: str

class RespostaRAG(BaseModel):
    answer: str
    source_documents: list[str] = []

router = APIRouter()

rag_chain = None
retriever = None

def format_docs(docs):
    """Formata documentos para o contexto."""
    print(f"Formatando {len(docs)} documentos recuperados")
    for i, doc in enumerate(docs):
        print(f"  Doc {i+1}: {doc.page_content[:100]}...")
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_pipeline():
    """Configura e retorna o pipeline RAG completo."""
    global rag_chain, retriever
    
    print("Iniciando a configuração do pipeline RAG...")
    
    if not CHROMA_PATH.exists():
        raise ValueError(
            f"Diretório ChromaDB não encontrado em: {CHROMA_PATH}\n"
            f"Execute 'python index_data.py' primeiro."
        )
    
    print("Carregando modelo de embedding...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Carregando ChromaDB de: {CHROMA_PATH}")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
        collection_name="compliance_docs"
    )
    
    collection = vectorstore._collection
    count = collection.count()
    print(f"ChromaDB carregado: {count} documentos encontrados")
    
    if count == 0:
        raise ValueError(
            f"Nenhum documento encontrado no ChromaDB!\n"
            f"Caminho verificado: {CHROMA_PATH}\n"
            f"Execute 'python index_data.py' para indexar os documentos."
        )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    print("Configurando Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente de compliance da Dunder Mifflin Paper Company.

Use APENAS o contexto fornecido abaixo para responder à pergunta.
Se a resposta não estiver no contexto, diga: "Não encontrei essa informação no manual de compliance."

CONTEXTO:
{context}"""),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Pipeline RAG configurado com sucesso.")
    return rag_chain, retriever

@router.get("/")
def read_root():
    """Endpoint raiz."""
    return {"status": "RAG API running. Use /chat endpoint."}

@router.post("/chat", response_model=RespostaRAG)
async def chat_endpoint(pergunta: Pergunta):
    """Recebe uma pergunta, executa o RAG e retorna a resposta."""
    global rag_chain, retriever
    
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline não inicializado.")
    
    try:
        print(f"\nNova pergunta: {pergunta.query}")
        
        docs = retriever.invoke(pergunta.query)
        print(f"Documentos recuperados: {len(docs)}")
        
        print("Gerando resposta com Gemini...")
        answer = rag_chain.invoke(pergunta.query)
        
        source_texts = [doc.page_content[:200] + "..." for doc in docs]
        
        print(f"Resposta gerada: {answer[:100]}...")
        
        return RespostaRAG(
            answer=answer,
            source_documents=source_texts
        )
        
    except Exception as e:
        print(f"Erro ao processar a consulta: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar a pergunta: {str(e)}"
        )

@router.get("/health")
def health_check():
    """Endpoint para verificar se o servidor está funcionando."""
    return {
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "chroma_path": str(CHROMA_PATH),
        "chroma_exists": CHROMA_PATH.exists()
    }

@router.get("/debug/test-retrieval")
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