# Chatbot - Parte 1

A solução é dividida em três camadas principais:

1.  **Camada de Serviço (FastAPI):** Atua como a **API de coordenação**, recebendo as requisições do usuário e orquestrando o fluxo RAG assíncrono.
2.  **Camada de Recuperação (Local):** Responsável por indexar e buscar informações nos documentos confidenciais.
3.  **Camada de Geração (Nuvem/API):** Utiliza a biblioteca `google-genai` para comunicação com o LLM (Gemini) na nuvem.

O fluxo de trabalho é dividido em duas fases principais: **Indexação** (offline) e **Consulta Híbrida** (em tempo real).

#### 1\. Indexação Local (Preparação de Dados)

O objetivo é transformar documentos em uma estrutura que possa ser consultada rapidamente:

1.  **Carregamento de Documentos:** O arquivo de origem (`seu_documento.pdf`, `.docx`, etc.) é carregado.
2.  **Chunking:** O documento é dividido em **pedaços (chunks)** menores.
3.  **Geração de Embeddings:** Um **Modelo de Embedding** transforma cada chunk em um **vetor numérico**.
4.  **Armazenamento:** Os chunks de texto originais e seus vetores correspondentes são armazenados no **ChromaDB** local.

#### 2\. Consulta Híbrida (Endpoint `/chat`)

Este é o fluxo que ocorre a cada requisição do usuário, orquestrado pelo FastAPI:

1.  **Requisição do Usuário:** O **FastAPI** recebe a pergunta do usuário através do endpoint `/chat`.
2.  **Busca Local (Retrieval):**
      * A pergunta do usuário é transformada em um vetor de consulta ($V_{query}$) usando o **mesmo Modelo de Embedding** da etapa de Indexação.
      * O **ChromaDB** busca os **$k$ vetores mais próximos** ($V_{rel}$) de $V_{query}$ (e seus respectivos chunks de texto) usando similaridade de cosseno. Estes são os **trechos relevantes locais**.
3.  **Montagem do Prompt (Aumento):** O código Python constrói o **prompt final** que será enviado à nuvem, estruturado da seguinte forma:
    > $$Prompt_{final} = [Instrução/System Prompt] + [Trechos Locais Encontrados] + [Pergunta do Usuário]$$
4.  **Geração na Nuvem (Generation):**
      * O `Prompt_{final}` é enviado ao **Modelo Gemini** (via `google-genai` API).
      * O LLM gera uma resposta contextualizada, usando as informações dos trechos locais como **base de conhecimento**.
5.  **Resposta:** A resposta gerada pelo Gemini, juntamente com a lista de documentos de origem (trechos utilizados), é formatada e enviada de volta ao usuário pelo FastAPI.