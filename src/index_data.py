"""
Script para indexação de documentos de compliance no ChromaDB
Versão simplificada com menos dependências
"""
import os
import shutil
from pathlib import Path

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("\n🔧 Execute os seguintes comandos para corrigir:")
    print("\npip uninstall langchain langchain-community langchain-core langchain-chroma -y")
    print("pip install langchain langchain-community langchain-text-splitters langchain-chroma chromadb sentence-transformers")
    exit(1)

def index_documents():
    """Indexa documentos de compliance no ChromaDB"""
    
    file_path = "data/politica_compliance.txt"
    
    # Verifica se o arquivo existe
    if not os.path.exists(file_path):
        print(f"❌ Erro: Arquivo '{file_path}' não encontrado!")
        print(f"📁 Caminho procurado: {os.path.abspath(file_path)}")
        
        # Cria um arquivo de exemplo se não existir
        os.makedirs("data", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("""
# Política de Compliance Empresarial

## 1. Código de Conduta
Todos os colaboradores devem agir com integridade e ética.

## 2. Prevenção à Lavagem de Dinheiro
A empresa adota políticas rigorosas de KYC (Know Your Customer).

## 3. Proteção de Dados
Seguimos a LGPD para proteção de dados pessoais.

## 4. Conflito de Interesses
Colaboradores devem declarar potenciais conflitos de interesse.

## 5. Canal de Denúncias
Disponível 24/7 para relatar irregularidades de forma anônima.
""")
        print(f"✓ Arquivo de exemplo criado em {file_path}")
    
    print("📄 Carregando documentos...")
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"✓ {len(documents)} documento(s) carregado(s)")
    except Exception as e:
        print(f"❌ Erro ao carregar documento: {e}")
        return
    
    print("\n✂️  Segmentando documentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ {len(chunks)} chunks criados")
    
    if len(chunks) == 0:
        print("⚠️  Nenhum chunk foi criado. Verifique o conteúdo do arquivo.")
        return
    
    print("\n🧠 Criando embeddings...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"❌ Erro ao carregar modelo de embeddings: {e}")
        return
    
    persist_dir = "./chroma_db"
    
    # Remove banco antigo se existir
    if os.path.exists(persist_dir):
        print(f"🗑️  Removendo banco antigo...")
        shutil.rmtree(persist_dir)
    
    os.makedirs(persist_dir, exist_ok=True)
    
    print("\n💾 Salvando no ChromaDB...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir,
            collection_name="compliance_docs"
        )
        
        print(f"\n✅ Indexação concluída com sucesso!")
        print(f"📊 Total de chunks: {len(chunks)}")
        print(f"💾 Banco salvo em: {os.path.abspath(persist_dir)}")
        
        # Teste de busca
        print("\n🔍 Testando busca...")
        results = vectorstore.similarity_search("compliance", k=2)
        print(f"✓ Busca funcionando! {len(results)} resultados encontrados")
        
        if results:
            print("\n📝 Exemplo de resultado:")
            print(f"   {results[0].page_content[:150]}...")
            
    except Exception as e:
        print(f"❌ Erro ao criar vectorstore: {e}")
        import traceback
        traceback.print_exc()

def load_existing_vectorstore():
    """Carrega um vectorstore existente"""
    
    persist_dir = "./chroma_db"
    
    if not os.path.exists(persist_dir):
        print(f"❌ Banco de dados não encontrado em {persist_dir}")
        return None
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name="compliance_docs"
    )
    
    print("✓ Vectorstore carregado com sucesso!")
    return vectorstore

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INDEXADOR DE DOCUMENTOS CHROMADB")
    print("=" * 60)
    
    try:
        index_documents()
    except Exception as e:
        print(f"\n❌ Erro durante a indexação: {str(e)}")
        print("\n🔧 Comandos para corrigir dependências:")
        print("\npip uninstall langchain langchain-community langchain-core -y")
        print("pip install langchain==0.1.20 langchain-community==0.0.38 langchain-chroma chromadb sentence-transformers")
        
        import traceback
        print("\n📋 Traceback completo:")
        traceback.print_exc()