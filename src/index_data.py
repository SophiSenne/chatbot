"""
Script para indexação de documentos de compliance no ChromaDB
Executar APENAS UMA VEZ para criar o banco de vetores
"""

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def index_documents():
    """Indexa documentos de compliance no ChromaDB"""
    
    # Verificar se o arquivo existe
    file_path = "data/politica_compliance.txt"
    if not os.path.exists(file_path):
        print(f"❌ Erro: Arquivo '{file_path}' não encontrado!")
        print("Certifique-se de criar o diretório 'data' e adicionar o arquivo.")
        return
    
    print("📄 Carregando documentos...")
    # --- 1. Carregar Dados ---
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"✓ {len(documents)} documento(s) carregado(s)")
    
    print("\n✂️  Segmentando documentos...")
    # --- 2. Segmentação (Chunking) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ {len(chunks)} chunks criados")
    
    print("\n🧠 Criando embeddings e salvando no ChromaDB...")
    # --- 3. Criar Embeddings e Salvar ---
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use 'cuda' se tiver GPU
    )
    
    # Criar diretório se não existir
    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)
    
    # Criar e persistir o vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"\n✅ Indexação concluída com sucesso!")
    print(f"📊 Total de chunks: {len(chunks)}")
    print(f"💾 Banco salvo em: {persist_dir}")
    print(f"\n🔍 Testando busca...")
    
    # Teste rápido
    results = vectorstore.similarity_search("compliance", k=2)
    print(f"✓ Busca funcionando! Encontrados {len(results)} resultados")

if __name__ == "__main__":
    try:
        index_documents()
    except Exception as e:
        print(f"\n❌ Erro durante a indexação: {str(e)}")
        print("\nVerifique se todos os pacotes estão instalados:")
        print("pip install langchain langchain-community chromadb sentence-transformers")