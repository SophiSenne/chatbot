import os
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY não encontrada! "
        "Configure a variável de ambiente ou crie um arquivo .env"
    )

class InvestigationResultLangChain(BaseModel):
    """Modelo estruturado para o resultado da investigação (LangChain)"""
    suspeita_veridica: bool = Field(
        description="Se a suspeita de conspiração é verdadeira (True) ou falsa (False)"
    )
    conclusao_justificada: str = Field(
        description="Explicação detalhada e fundamentada da conclusão alcançada"
    )
    evidencias_encontradas: List[str] = Field(
        description="Lista de citações exatas dos e-mails que servem como evidências"
    )

class InvestigationRequest(BaseModel):
    """Modelo de requisição para investigação"""
    emails_content: str = Field(
        ..., 
        description="Conteúdo dos e-mails a serem analisados",
        min_length=10
    )
    suspeito: str = Field(
        default="Michael Scott",
        description="Nome do suspeito"
    )
    alvo: str = Field(
        default="Toby Flenderson",
        description="Nome do alvo"
    )
    suspeita: str = Field(
        default="Conspiração ativa",
        description="Descrição da suspeita"
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Modelo Gemini a ser usado"
    )


class InvestigationResponse(BaseModel):
    """Modelo de resposta da investigação"""
    suspeita_veridica: bool = Field(
        ..., 
        description="Se a suspeita foi confirmada"
    )
    conclusao_justificada: str = Field(
        ..., 
        description="Conclusão detalhada da investigação"
    )
    evidencias_encontradas: List[str] = Field(
        ..., 
        description="Lista de evidências encontradas"
    )
    status: str = Field(
        ..., 
        description="Status da investigação (CONFIRMADA ou NÃO CONFIRMADA)"
    )
    total_evidencias: int = Field(
        ..., 
        description="Número total de evidências encontradas"
    )

router = APIRouter(
    prefix="/audit",
    tags=["Auditoria"],
    responses={404: {"description": "Not found"}},
)


def create_investigation_chain(model_name: str = "gemini-2.0-flash"):
    """
    Cria a chain de investigação com Gemini LLM e parser estruturado.
    """

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=GEMINI_API_KEY,
        convert_system_message_to_human=True
    )
    
    parser = JsonOutputParser(pydantic_object=InvestigationResultLangChain)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Você é um **Agente de Investigação Corporativa Especializado** 
        com expertise em análise forense de comunicações internas.
        
        Sua missão é investigar com rigor e imparcialidade se há evidências de 
        conspiração, planejamento hostil ou comunicação secreta prejudicial.
        
        **CRITÉRIOS DE ANÁLISE:**
        - Procure por padrões de comportamento hostil sistemático
        - Identifique planejamentos ou ações coordenadas contra indivíduos
        - Detecte linguagem conspiratória ou comunicação secreta
        - Diferencie piadas/conflitos normais de conspiração real
        - Avalie o contexto organizacional e relações interpessoais
        
        **IMPORTANTE:** Seja objetivo e baseie-se apenas em evidências concretas.
        Citações devem ser EXATAS do texto original.
        
        {format_instructions}"""),
        
        ("human", """**CASO DE INVESTIGAÇÃO:**
        
        **Suspeito:** {suspeito}
        **Alvo:** {alvo}
        **Suspeita:** {suspeita}
        
        **E-MAILS PARA ANÁLISE:**
        
        {emails_content}
        
        ---
        
        Analise minuciosamente os e-mails acima e determine se {suspeito} 
        está conspirando contra {alvo}. Forneça sua conclusão em formato JSON.""")
    ])
    
    chain = prompt_template | llm | parser
    
    return chain, parser


async def perform_investigation(
    emails_content: str,
    suspeito: str,
    alvo: str,
    suspeita: str,
    model_name: str
) -> dict:
    """
    Executa a investigação sobre os e-mails.
    """
    try:
        chain, parser = create_investigation_chain(model_name)
        
        result = chain.invoke({
            "emails_content": emails_content,
            "suspeito": suspeito,
            "alvo": alvo,
            "suspeita": suspeita,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a investigação: {str(e)}"
        )

@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    summary="Investigar conspiração a partir de arquivo",
    description="Carrega e-mails de um arquivo .txt e realiza a investigação"
)
async def investigate_from_file(
    file: UploadFile = File(..., description="Arquivo .txt contendo os e-mails"),
    suspeito: str = "Michael Scott",
    alvo: str = "Toby Flenderson",
    suspeita: str = "Conspiração ativa",
    model_name: str = "gemini-2.0-flash"
):
    try:

        if not file.filename.endswith('.txt'):
            raise HTTPException(
                status_code=400,
                detail="Apenas arquivos .txt são aceitos"
            )
        
        content = await file.read()
        emails_content = content.decode('utf-8')
        
        if len(emails_content) < 10:
            raise HTTPException(
                status_code=400,
                detail="Arquivo muito pequeno ou vazio"
            )
        
        result = await perform_investigation(
            emails_content=emails_content,
            suspeito=suspeito,
            alvo=alvo,
            suspeita=suspeita,
            model_name=model_name
        )
        
        status = "CONFIRMADA" if result['suspeita_veridica'] else "NÃO CONFIRMADA"
        
        return InvestigationResponse(
            suspeita_veridica=result['suspeita_veridica'],
            conclusao_justificada=result['conclusao_justificada'],
            evidencias_encontradas=result['evidencias_encontradas'],
            status=status,
            total_evidencias=len(result['evidencias_encontradas'])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar arquivo: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check do serviço de auditoria",
    description="Verifica se o serviço está funcionando corretamente"
)
async def health_check():
    """
    Endpoint de health check.
    """
    return {
        "status": "healthy",
        "service": "Audit Investigation Agent",
        "gemini_configured": bool(GEMINI_API_KEY)
    }