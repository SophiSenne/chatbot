from __future__ import annotations

import os
import csv
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    # Fallback se google-api-core não estiver disponível
    class ResourceExhausted(Exception):
        """Exceção para quota excedida do Gemini"""
        pass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "⚠️ GEMINI_API_KEY não encontrada! "
        "Configure a variável de ambiente ou crie um arquivo .env"
    )

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DATA_DIR = (PROJECT_DIR / "data").resolve()
EMAIL_DB_DIR = (PROJECT_DIR / "chroma_emails").resolve()

COMPLIANCE_PATH = DATA_DIR / "politica_compliance.txt"
TRANSACTIONS_PATH = DATA_DIR / "transacoes_bancarias.csv"
EMAILS_PATH = DATA_DIR / "emails.txt"

router = APIRouter(prefix="/fraud", tags=["fraud"])

compliance_text: Optional[str] = None
transactions_cache: Optional[List[dict]] = None
email_retriever = None
llm: Optional[ChatGoogleGenerativeAI] = None


class FraudScanRequest(BaseModel):
    """Configurações para a varredura de fraudes."""

    limit: Optional[int] = Field(
        default=None,
        description="Limita o número de transações analisadas (para testes rápidos).",
    )
    contextual: bool = Field(
        default=True,
        description="Se True, faz checagem contextual com emails.",
    )
    email_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número de trechos de email recuperados por transação.",
    )
    rebuild_email_index: bool = Field(
        default=False,
        description="Força recriar o índice vetorial de emails.",
    )


class TransactionFlag(BaseModel):
    """Resultado de avaliação de uma transação."""

    transaction: dict
    violation: bool
    severity: str
    reason: str
    matched_rules: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None


class FraudScanResponse(BaseModel):
    direct: List[TransactionFlag]
    contextual: List[TransactionFlag]


def _ensure_files_exist():
    missing = [
        str(path.name)
        for path in [COMPLIANCE_PATH, TRANSACTIONS_PATH, EMAILS_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Arquivos ausentes em '{DATA_DIR}': {', '.join(missing)}"
        )


def _load_transactions() -> List[dict]:
    with TRANSACTIONS_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        raise ValueError("Nenhuma transação encontrada em transacoes_bancarias.csv")

    return rows


def _build_email_retriever(rebuild: bool = False):
    if rebuild and EMAIL_DB_DIR.exists():
        shutil.rmtree(EMAIL_DB_DIR)

    if EMAIL_DB_DIR.exists():
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        vectorstore = Chroma(
            persist_directory=str(EMAIL_DB_DIR),
            embedding_function=embedding,
            collection_name="email_chunks",
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    email_text = EMAILS_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(email_text)

    if not chunks:
        raise ValueError("Nenhum chunk de email criado; verifique emails.txt")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory=str(EMAIL_DB_DIR),
        collection_name="email_chunks",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def setup_fraud_pipeline(rebuild_email_index: bool = False):
    """Inicializa modelos e caches do agente de fraude."""
    global compliance_text, transactions_cache, email_retriever, llm

    _ensure_files_exist()

    compliance_text = COMPLIANCE_PATH.read_text(encoding="utf-8")
    transactions_cache = _load_transactions()
    email_retriever = _build_email_retriever(rebuild=rebuild_email_index)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1, google_api_key=GEMINI_API_KEY,
        convert_system_message_to_human=True)

    return {
        "transactions_loaded": len(transactions_cache),
        "emails_indexed": True,
        "policy_chars": len(compliance_text),
    }


def _require_pipeline():
    if not all([compliance_text, transactions_cache, email_retriever, llm]):
        raise HTTPException(
            status_code=500,
            detail="Pipeline de fraude não inicializado. "
            "Chame setup_fraud_pipeline() ou acesse /fraud/health para detalhes.",
        )


class _DirectVerdict(BaseModel):
    violation: bool
    severity: str
    justification: str
    matched_rules: List[str]
    recommended_action: str


class _ContextVerdict(BaseModel):
    violation: bool
    severity: str
    justification: str
    matched_rules: List[str]
    evidence: List[str]
    recommended_action: str


def _evaluate_direct(transaction: dict) -> TransactionFlag:
    assert compliance_text is not None and llm is not None

    structured_llm = llm.with_structured_output(_DirectVerdict)
    prompt = ChatPromptTemplate.from_template(
        (
            "Você é um auditor de compliance. Use apenas a política abaixo para "
            "classificar a transação como 'violation' True/False. "
            "Inclua referências específicas às regras que foram violadas.\n\n"
            "POLÍTICA DE COMPLIANCE:\n{policy}\n\n"
            "TRANSAÇÃO:\n{transaction}\n"
        )
    )

    chain = prompt | structured_llm
    verdict = chain.invoke(
        {
            "policy": compliance_text,
            "transaction": json.dumps(transaction, ensure_ascii=False),
        }
    )

    return TransactionFlag(
        transaction=transaction,
        violation=verdict.violation,
        severity=verdict.severity,
        reason=verdict.justification,
        matched_rules=verdict.matched_rules,
        recommended_action=verdict.recommended_action,
    )


def _evaluate_contextual(transaction: dict, email_k: int) -> TransactionFlag:
    assert compliance_text is not None and email_retriever is not None and llm is not None

    query = f"{transaction.get('funcionario','')} {transaction.get('descricao','')} {transaction.get('valor','')}"
    docs = email_retriever.invoke(query)
    top_emails = [doc.page_content for doc in docs[:email_k]]

    structured_llm = llm.with_structured_output(_ContextVerdict)
    prompt = ChatPromptTemplate.from_template(
        (
            "Você é um auditor investigando fraude que depende de contexto de comunicação. "
            "Use os emails fornecidos para decidir se há conluio ou tentativa de burlar "
            "as regras de compliance. Responda se é violação e dê a justificativa curta.\n\n"
            "POLÍTICA DE COMPLIANCE:\n{policy}\n\n"
            "TRANSAÇÃO:\n{transaction}\n\n"
            "TRECHOS DE EMAIL RELACIONADOS:\n{emails}\n"
        )
    )
    chain = prompt | structured_llm
    verdict = chain.invoke(
        {
            "policy": compliance_text,
            "transaction": json.dumps(transaction, ensure_ascii=False),
            "emails": "\n---\n".join(top_emails),
        }
    )

    return TransactionFlag(
        transaction=transaction,
        violation=verdict.violation,
        severity=verdict.severity,
        reason=verdict.justification,
        matched_rules=verdict.matched_rules,
        evidence=verdict.evidence or top_emails,
        recommended_action=verdict.recommended_action,
    )


@router.get("/health")
def fraud_health():
    """Retorna o status do agente de fraude."""
    status = {
        "compliance_loaded": bool(compliance_text),
        "transactions_loaded": bool(transactions_cache),
        "email_index_built": email_retriever is not None and EMAIL_DB_DIR.exists(),
        "data_dir": str(DATA_DIR),
    }
    return status


@router.post("/setup")
def fraud_setup(cfg: FraudScanRequest):
    """Inicializa o pipeline de fraude sob demanda."""
    try:
        summary = setup_fraud_pipeline(rebuild_email_index=cfg.rebuild_email_index)
        return {"status": "ready", **summary}
    except Exception as exc:  # pragma: no cover - FastAPI surface
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/scan", response_model=FraudScanResponse)
def fraud_scan(cfg: FraudScanRequest):
    """Executa a checagem de fraudes (direta e contextual)."""
    if not all([compliance_text, transactions_cache, email_retriever, llm]):
        try:
            setup_fraud_pipeline(rebuild_email_index=cfg.rebuild_email_index)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    _require_pipeline()

    transactions = transactions_cache or []

    # -----------------------------
    # Heurística simples de "suspeição"
    # -----------------------------
    KEYWORDS = [
        "reembolso", "reemb", "presente", "gift", "cartão", "cartao",
        "viagem", "hotel", "restaurante", "jantar", "almoço", "almoco",
        "entretenimento", "cash", "dinheiro", "pix", "transfer",
        "consultoria", "consulting", "propina", "comissão", "comissao",
        "fornecedor", "supplier", "patrocínio", "patrocinio"
    ]

    def _parse_brl(valor_raw) -> float:
        if valor_raw is None:
            return 0.0
        s = str(valor_raw).strip()
        # aceita "82.84", "82,84", "R$ 82,84"
        s = s.replace("R$", "").replace(" ", "")
        if "," in s and "." in s:
            # se vier "1.234,56" -> remove milhar e troca decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    def _suspicion_score(tx: dict) -> float:
        desc = str(tx.get("descricao", "") or "").lower()
        func = str(tx.get("funcionario", "") or "").lower()
        valor = _parse_brl(tx.get("valor"))

        score = 0.0
        # valor alto pesa mais
        score += min(valor / 200.0, 20.0)  # controla explosão

        # palavras-chave
        for kw in KEYWORDS:
            if kw in desc:
                score += 3.0

        # alguns nomes “problemáticos” (opcional, ajuda na demo Office)
        for name in ["michael", "dwight", "andy", "ryan"]:
            if name in func:
                score += 1.0

        return score

    # Ordena do mais suspeito para o menos suspeito
    transactions = sorted(transactions, key=_suspicion_score, reverse=True)

    # -----------------------------
    # LIMITADOR SAFE MODE (quota)
    # -----------------------------
    SAFE_MAX_CONTEXTUAL = 1     # contextual = 2 chamadas/tx -> 1 é o mais estável
    SAFE_MAX_DIRECT_ONLY = 3

    requested_limit = cfg.limit if (cfg.limit and cfg.limit > 0) else None
    if cfg.contextual:
        effective_limit = min(requested_limit or SAFE_MAX_CONTEXTUAL, SAFE_MAX_CONTEXTUAL)
    else:
        effective_limit = min(requested_limit or SAFE_MAX_DIRECT_ONLY, SAFE_MAX_DIRECT_ONLY)

    transactions = transactions[:effective_limit]

    THROTTLE_SECONDS = 1.2

    direct_flags: List[TransactionFlag] = []
    contextual_flags: List[TransactionFlag] = []

    try:
        for idx, tx in enumerate(transactions, start=1):
            print(
                f"[fraud] analisando {idx}/{len(transactions)} | "
                f"score={_suspicion_score(tx):.2f} | "
                f"funcionario={tx.get('funcionario')} | valor={tx.get('valor')} | desc={tx.get('descricao')}"
            )

            direct_verdict = _evaluate_direct(tx)
            if direct_verdict.violation:
                direct_flags.append(direct_verdict)

            if cfg.contextual:
                ctx_verdict = _evaluate_contextual(tx, cfg.email_k)
                if ctx_verdict.violation:
                    contextual_flags.append(ctx_verdict)

            if idx < len(transactions):
                time.sleep(THROTTLE_SECONDS)

    except ResourceExhausted as e:
        raise HTTPException(
            status_code=429,
            detail=(
                "Quota do Gemini excedida (429). "
                "Aguarde ~60s e tente novamente. "
                "Se estiver usando análise contextual, reduza o limite de transações "
                "ou desative o contextual."
            ),
        ) from e

    return FraudScanResponse(direct=direct_flags, contextual=contextual_flags)
