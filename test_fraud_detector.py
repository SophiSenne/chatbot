"""Script de teste para verificar o agente de fraude."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agents.fraud_detector import (
        DATA_DIR,
        COMPLIANCE_PATH,
        TRANSACTIONS_PATH,
        EMAILS_PATH,
        EMAIL_DB_DIR,
        setup_fraud_pipeline,
    )
    
    print("=" * 60)
    print("VERIFICAÇÃO DE CAMINHOS E ARQUIVOS")
    print("=" * 60)
    print(f"\nDATA_DIR: {DATA_DIR}")
    print(f"  Existe: {DATA_DIR.exists()}")
    
    print(f"\nCOMPLIANCE_PATH: {COMPLIANCE_PATH}")
    print(f"  Existe: {COMPLIANCE_PATH.exists()}")
    if COMPLIANCE_PATH.exists():
        size = COMPLIANCE_PATH.stat().st_size
        print(f"  Tamanho: {size} bytes")
    
    print(f"\nTRANSACTIONS_PATH: {TRANSACTIONS_PATH}")
    print(f"  Existe: {TRANSACTIONS_PATH.exists()}")
    if TRANSACTIONS_PATH.exists():
        size = TRANSACTIONS_PATH.stat().st_size
        print(f"  Tamanho: {size} bytes")
    
    print(f"\nEMAILS_PATH: {EMAILS_PATH}")
    print(f"  Existe: {EMAILS_PATH.exists()}")
    if EMAILS_PATH.exists():
        size = EMAILS_PATH.stat().st_size
        print(f"  Tamanho: {size} bytes")
    
    print(f"\nEMAIL_DB_DIR: {EMAIL_DB_DIR}")
    print(f"  Existe: {EMAIL_DB_DIR.exists()}")
    
    print("\n" + "=" * 60)
    print("TESTANDO SETUP DO PIPELINE")
    print("=" * 60)
    
    try:
        result = setup_fraud_pipeline(rebuild_email_index=False)
        print("\n✅ Pipeline inicializado com sucesso!")
        print(f"  Transações carregadas: {result['transactions_loaded']}")
        print(f"  Política de compliance: {result['policy_chars']} caracteres")
        print(f"  Índice de emails: {'Sim' if result['emails_indexed'] else 'Não'}")
    except Exception as e:
        print(f"\n❌ Erro ao inicializar pipeline: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("\nTente instalar as dependências:")
    print("  pip install langchain-huggingface langchain-chroma langchain-text-splitters")
    import traceback
    traceback.print_exc()