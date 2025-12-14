import os
import shutil
from pathlib import Path

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError as e:
    print(f"Erro: {e}")
    exit(1)

def index_documents():
    # Usa caminhos absolutos baseados na localização do script
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]  # Volta para a raiz do projeto
    data_dir = project_dir / "data"
    file_path = data_dir / "politica_compliance.txt"
    
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo '{file_path}' não encontrado")
        print(f"Caminho procurado: {os.path.abspath(file_path)}")
    
    print("Carregando documentos")
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"{len(documents)} documento(s) carregado(s)")
    except Exception as e:
        print(f"Erro ao carregar documento: {e}")
        return
    
    print("\nSegmentando documentos")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} chunks criados")
    
    if len(chunks) == 0:
        print("Nenhum chunk foi criado. Verifique o conteúdo do arquivo.")
        return
    
    print("\nCriando embeddings")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"Erro ao carregar modelo de embeddings: {e}")
        return
    
    # Usa caminho relativo ao src/ para manter compatibilidade com chatbot.py
    persist_dir = script_dir.parent / "chroma_db"
    
    if persist_dir.exists():
        print(f"Removendo banco antigo")
        shutil.rmtree(persist_dir)
    
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSalvando no ChromaDB")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(persist_dir),
            collection_name="compliance_docs"
        )
        
        print(f"\nIndexação concluída com sucesso!")
        print(f"Total de chunks: {len(chunks)}")
        print(f"Banco salvo em: {persist_dir.resolve()}")
        
        print("\nTestando busca...")
        results = vectorstore.similarity_search("compliance", k=2)
        print(f"Busca funcionando! {len(results)} resultados encontrados")
        
        if results:
            print("\nExemplo de resultado:")
            print(f"   {results[0].page_content[:150]}...")
            
    except Exception as e:
        print(f"Erro ao criar vectorstore: {e}")
        import traceback
        traceback.print_exc()

def load_existing_vectorstore():
    script_dir = Path(__file__).resolve().parent
    persist_dir = script_dir.parent / "chroma_db"
    
    if not persist_dir.exists():
        print(f"Banco de dados não encontrado em {persist_dir}")
        return None
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_name="compliance_docs"
    )
    
    print("Vectorstore carregado com sucesso!")
    return vectorstore

if __name__ == "__main__":   
    try:
        index_documents()
    except Exception as e:
        print(f"\nErro: {str(e)}")
        
        import traceback
        print("\nTraceback completo:")
        traceback.print_exc()