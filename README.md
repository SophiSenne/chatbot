## Chatbot de Auditoria da Dunder Mifflin (Agente Inteligente)

Sistema de agentes inteligentes para automatizar a auditoria de compliance e investigações de e-mails, transformando o volume de documentos em conhecimento estruturado e acionável.

### Demonstração e Uso (Link do Vídeo)

O vídeo abaixo demonstra o sistema em funcionamento, cobrindo os requisitos de auditoria e investigação:

➡️ **[LINK PARA O VÍDEO EXPLICATIVO (COLOCAR AQUI)](https://www.google.com/search?q=https://www.youtube.com/link_da_sua_demostracao)**

### Arquitetura

O sistema é modular, dividido em **três agentes de IA** e uma **API de orquestração (FastAPI)**. O coração do sistema é a arquitetura RAG (Retrieval-Augmented Generation) para garantir que as respostas sejam factuais e baseadas em seus documentos.

#### Desenho dos Agentes

O projeto utiliza o **FastAPI** como a camada de serviço principal e **LangChain** para orquestrar o fluxo de trabalho dos Agentes.

| Agente / Componente | Propósito Principal | Dados de Entrada |
| :--- | :--- | :--- |
| **1. Agente RAG (Chatbot de Compliance)** | Responde a dúvidas gerais dos colaboradores sobre a `politica_compliance.txt` (Requisito 1). | $V_{query}$ (Pergunta do usuário) |
| **2. Agente de Investigação (E-mails)** | Analisa o `emails_internos.txt` para identificar padrões de hostilidade, conspiração e fraude (Requisito 2 e 3.2). | E-mails e parâmetros (suspeito/alvo) |
| **3. Agente de Compliance (Transações)** | Valida transações do `transacoes_bancarias.csv` contra as regras da `politica_compliance.txt` (Requisito 3.1). | Transações CSV + Regras RAG |
| **API de Orquestração** | Expõe os Agentes via endpoints REST e coordena fluxos de trabalho. | Requisições HTTP |

### Instruções de Como Rodar o Projeto

Siga os passos abaixo para preparar e executar o ambiente de desenvolvimento local.

#### 1\. Configuração do Ambiente

Crie um ambiente virtual e ative-o:

```bash
python -m venv venv
source venv/bin/activate  # No Windows use: .\venv\Scripts\activate
```

#### 2\. Instalação de Dependências

Instale todas as bibliotecas necessárias listadas no seu `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### 3\. Configuração da Chave de API (Obrigatório)

O sistema depende do acesso ao modelo Gemini (Google AI).

1.  Obtenha sua chave de API no Google AI Studio.
2.  Crie um arquivo chamado `.env` na raiz do projeto.
3.  Adicione sua chave a este arquivo:

<!-- end list -->

```
# .env
GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

> **Atenção:** O arquivo `.env` está no `.gitignore` e **não deve ser "commitado"** no repositório.

#### 4\. Preparação dos Dados (Indexação RAG)

Antes de rodar a API, você deve indexar a `politica_compliance.txt` no seu Vector Store local (**ChromaDB**). Este processo prepara o **Agente RAG**.

Execute o script de indexação:

```bash
python3 src/scripts/index_data.py
```

#### 5\. Execução da API

Com o ambiente configurado e os dados indexados, inicie a API do FastAPI:

```bash
python3 src/main.py
```

O serviço estará disponível em: **`http://127.0.0.1:8000`**

Você pode interagir com os endpoints usando a documentação interativa: **`http://127.0.0.1:8000/docs`**

> **Dica:** Um arquivo **HTML de teste** para o frontend está disponível em `src/frontend/index.html` e pode ser usado para uma interface de teste amigável.

### Endpoints Principais para Teste

Use a interface `/docs` (Swagger UI) para testar os requisitos de Toby, navegando pelos roteadores de chat, fraude e auditoria.

| Método | Endpoint | Descrição |
| :--- | :--- | :--- |
| **POST** | `/chat` | Enviar pergunta ao Chatbot RAG. |
| **GET** | `/debug/test-retrieval` | Testar recuperação de documentos. | Testa busca vetorial com query de exemplo. |
| **POST** | `/fraud/setup` | Inicializar pipeline de fraude manualmente. Body: `FraudScanRequest` (limit, contextual, email\_k, rebuild\_email\_index). |
| **POST** | `/fraud/scan` | Executar varredura de fraudes (direta e contextual) |
| **POST** | `/audit/investigate` | Investigar conspiração via upload de arquivo de e-mails (Requisito 2). Tipo: `multipart/form-data`. |

