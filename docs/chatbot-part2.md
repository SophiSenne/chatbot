# Documentação Ponderada ChatBot - Parte 2

## 1. Visão Geral do Projeto/Módulo

### Propósito Principal

O agente de audição é um microserviço REST API desenvolvido em Python que utiliza Inteligência Artificial Generativa (Google Gemini) para realizar análises automatizadas de e-mails corporativos internos. O sistema identifica padrões de comportamento hostil, conspirações ou comunicações prejudiciais coordenadas, retornando conclusões estruturadas com evidências extraídas dos textos analisados.

### Funcionalidade Geral

- **Análise de E-mails via Upload**: Recebe arquivos `.txt` contendo e-mails e processa através de IA
- **Investigação Estruturada**: Utiliza LangChain + Gemini LLM para análise contextual profunda
- **Respostas Validadas**: Retorna JSON estruturado com conclusões, evidências e status da investigação
- **Health Check**: Endpoint para monitoramento da saúde do serviço

### Tecnologias Utilizadas
- **FastAPI**: Framework web assíncrono para criação da API REST
- **LangChain**: Orquestração de pipelines de IA e processamento de linguagem natural
- **Google Gemini AI**: Modelo de linguagem para análise semântica dos e-mails
- **Pydantic**: Validação automática de dados de entrada e saída


## 2. Endpoints HTTP (API Layer)

Esta camada expõe a funcionalidade via HTTP REST.

### 2.1 `POST /audit/investigate`

**Responsabilidade**: Análise via upload de arquivo

**Método HTTP**: `POST`

**Exemplo de Requisição**:
```bash
curl -X POST "http://localhost:8000/audit/investigate" \
  -H "accept: application/json" \
  -F "file=@emails.txt" \
  -F "suspeito=John Doe" \
  -F "alvo=Jane Smith"
```

**Resposta esperada**:
```json
{
  "suspeita_veridica": true,
  "conclusao_justificada": "Análise dos e-mails revela comunicação coordenada entre Michael Scott e Dwight Schrute com clara intenção de prejudicar Toby Flenderson. Observa-se linguagem conspiratória ('Plano Confidencial'), planejamento de reunião secreta, e preparação de dossiê com 'evidências contra ele'.",
  "evidencias_encontradas": [
    "Dwight, precisamos conversar sobre o Toby. Ele está atrapalhando nossos planos novamente.",
    "Vamos nos reunir às 15h na sala de conferências para discutir como proceder.",
    "Vou preparar o dossiê com todas as evidências contra ele. Desta vez ele não escapa."
  ],
  "status": "CONFIRMADA",
  "total_evidencias": 3
}

```

### 2.2 `GET /audit/health`

**Responsabilidade**: Health check do serviço

**Método HTTP**: `GET`

**Exemplo de Requisição**:

```bash
curl -X GET "http://localhost:8000/audit/health"
```

**Resposta esperada**:
```json
{
  "status": "healthy",
  "service": "Audit Investigation Agent",
  "gemini_configured": true
}
```

## 3. Fluxo de Dados Detalhado

O diagrama a seguir ilustra o fluxo de dados completo, desde a requisição HTTP até o processamento pelo LLM e a resposta final.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AUDIT INVESTIGATION AGENT                       │
│                         Fluxo Completo de Dados                          │
└─────────────────────────────────────────────────────────────────────────┘

1. ENTRADA
   ┌──────────────────┐
   │  Cliente HTTP    │
   │  (Browser/cURL)  │
   └────────┬─────────┘
            │ POST /audit/investigate
            │ {file: emails.txt, suspeito: "X", alvo: "Y"}
            ↓

2. VALIDAÇÃO DE ENTRADA
   ┌──────────────────────────────────────┐
   │     FastAPI Router                   │
   │  ┌────────────────────────────────┐  │
   │  │  Pydantic Request Validation   │  │
   │  │  • Validar tipos               │  │
   │  │  • Validar tamanho mínimo      │  │
   │  │  • Validar extensão .txt       │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ Dados validados
                  ↓

3. ORQUESTRAÇÃO
   ┌──────────────────────────────────────┐
   │   perform_investigation()            │
   │  ┌────────────────────────────────┐  │
   │  │  1. Criar chain                │  │
   │  │  2. Preparar parâmetros        │  │
   │  │  3. Invocar pipeline           │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ Parâmetros preparados
                  ↓

4. CRIAÇÃO DO PIPELINE
   ┌──────────────────────────────────────┐
   │   create_investigation_chain()       │
   │  ┌────────────────────────────────┐  │
   │  │  LLM: ChatGoogleGenerativeAI   │  │
   │  │  Parser: JsonOutputParser      │  │
   │  │  Prompt: ChatPromptTemplate    │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ Chain montada
                  ↓

5. PROCESSAMENTO LLM
   ┌──────────────────────────────────────┐
   │        Gemini LLM (Google)           │
   │  ┌────────────────────────────────┐  │
   │  │  Prompt formatado:             │  │
   │  │  "Você é um investigador...    │  │
   │  │   Analise: [emails_content]"   │  │
   │  │                                │  │
   │  │  → Processamento semântico     │  │
   │  │  → Identificação de padrões    │  │
   │  │  → Extração de evidências      │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ Resposta do LLM (texto)
                  ↓

6. PARSING E ESTRUTURAÇÃO
   ┌──────────────────────────────────────┐
   │       JSON Parser                    │
   │  ┌────────────────────────────────┐  │
   │  │  Texto do LLM →                │  │
   │  │  {                             │  │
   │  │    "suspeita_veridica": true,  │  │
   │  │    "conclusao_justificada":...,│  │
   │  │    "evidencias_encontradas":[..]│ │
   │  │  }                             │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ JSON estruturado
                  ↓

7. ENRIQUECIMENTO DE DADOS
   ┌──────────────────────────────────────┐
   │   Formatação de Resposta             │
   │  ┌────────────────────────────────┐  │
   │  │  + Adicionar campo "status"    │  │
   │  │  + Calcular "total_evidencias" │  │
   │  │  + Validar com Pydantic        │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ InvestigationResponse
                  ↓

8. RESPOSTA HTTP
   ┌──────────────────────────────────────┐
   │     FastAPI Response                 │
   │  ┌────────────────────────────────┐  │
   │  │  Status: 200 OK                │  │
   │  │  Content-Type: application/json│  │
   │  │  Body: {                       │  │
   │  │    "suspeita_veridica": true,  │  │
   │  │    "status": "CONFIRMADA",     │  │
   │  │    ...                         │  │
   │  │  }                             │  │
   │  └────────────────────────────────┘  │
   └──────────────┬───────────────────────┘
                  │ HTTP Response
                  ↓
   ┌──────────────────┐
   │  Cliente HTTP    │
   │  (recebe JSON)   │
   └──────────────────┘
```

---

