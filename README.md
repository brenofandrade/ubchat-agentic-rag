# UBChat Agentic RAG

Sistema de RAG (Retrieval-Augmented Generation) com roteamento inteligente de consultas usando **Ollama** (modelos locais) e **Pinecone** (banco de dados vetorizado em nuvem).

## 🚀 Características

- **Modelos Locais**: Usa Ollama para executar LLMs localmente (economia de custos)
- **Vector Store em Nuvem**: Pinecone para armazenamento escalável de embeddings
- **Roteamento Inteligente**: Decide automaticamente entre RAG, resposta direta ou pedido de esclarecimento
- **Reranking Opcional**: Cross-encoder para melhorar relevância dos documentos
- **API REST**: Endpoints Flask para fácil integração
- **Histórico de Conversa**: Suporte a contexto conversacional

## 📋 Pré-requisitos

### 1. Ollama

Instale o Ollama seguindo as instruções em [ollama.ai](https://ollama.ai)

Baixe os modelos necessários:

```bash
# Modelo para geração de respostas
ollama pull llama3.2:latest

# Modelo para embeddings
ollama pull mxbai-embed-large:latest
```

Verifique se o Ollama está rodando:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### 2. Pinecone

1. Crie uma conta em [Pinecone](https://www.pinecone.io/)
2. Crie um índice com as seguintes configurações:
   - **Dimensões**: 1024 (para `mxbai-embed-large`)
   - **Métrica**: cosine
   - **Cloud**: Escolha a região mais próxima

3. Obtenha sua API Key no dashboard

## 🛠️ Instalação

### 1. Clone o repositório

```bash
git clone <repository-url>
cd ubchat-agentic-rag
```

### 2. Crie ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale dependências

```bash
pip install -r requirements.txt
```

### 4. Configure variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas configurações:

```bash
# Pinecone (obrigatório)
PINECONE_API_KEY_DSUNIBLU=your-pinecone-api-key
PINECONE_INDEX=your-index-name

# Ollama (ajuste se necessário)
OLLAMA_BASE_URL=http://localhost:11434
GENERATION_MODEL=llama3.2:latest
EMBEDDING_MODEL=mxbai-embed-large:latest
```

## 🚀 Executando o Sistema

### Iniciar o backend

```bash
python main.py
```

O servidor backend estará disponível em `http://localhost:8000`

### Iniciar a interface web (Streamlit)

Em outro terminal, execute:

```bash
streamlit run ui_app.py
```

A interface web estará disponível em `http://localhost:8501`

**Nota**: O backend deve estar rodando antes de iniciar a interface web.

### Health Check

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "provider": "ollama",
  "model": "llama3.2:latest",
  "pinecone_index": "your-index-name",
  "namespace": "default"
}
```

## 📡 API Endpoints

### 1. Roteamento de Consulta

**POST** `/route-query`

Decide a melhor estratégia para responder uma pergunta.

```bash
curl -X POST http://localhost:8000/route-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual é a política de férias da empresa?",
    "context": "Preciso saber sobre benefícios"
  }'
```

Resposta:

```json
{
  "route": "rag",
  "confidence": 0.95,
  "reasoning": "Pergunta sobre política interna da empresa",
  "suggested_documents": ["company_policies"]
}
```

### 2. Query RAG Completa

**POST** `/rag/query`

Recupera documentos e gera resposta.

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como funciona o processo de reembolso?",
    "top_k": 3,
    "chat_history": [
      {"role": "user", "content": "Oi"},
      {"role": "assistant", "content": "Olá! Como posso ajudar?"}
    ]
  }'
```

Resposta:

```json
{
  "answer": "De acordo com o Documento 1, o processo de reembolso...",
  "documents": [
    {
      "content": "Processo de Reembolso: ...",
      "metadata": {"source": "manual.pdf", "page": 5},
      "score": 0.92
    }
  ],
  "metadata": {
    "retrieved_count": 3,
    "generation_model": "llama3.2:latest",
    "embedding_model": "mxbai-embed-large:latest",
    "namespace": "default"
  }
}
```

### 3. Apenas Recuperação de Documentos

**POST** `/rag/retrieve`

Recupera documentos sem gerar resposta.

```bash
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "política de férias",
    "top_k": 5
  }'
```

### 4. Chat Completo (Roteamento + RAG)

**POST** `/chat`

Endpoint completo que decide automaticamente a melhor estratégia.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como funciona fotossíntese?",
    "chat_history": []
  }'
```

## 💻 Interface Web (Streamlit)

A aplicação inclui uma interface web moderna e intuitiva construída com Streamlit.

### Funcionalidades da Interface

- **Chat Interativo**: Interface de conversação natural
- **Histórico de Mensagens**: Mantém contexto da conversa
- **Exibição de Fontes**: Mostra documentos que foram usados para gerar a resposta
- **Verificação de Servidor**: Botão para verificar status do backend
- **Feedback**: Sistema de avaliação de respostas
- **Nova Conversa**: Botão para reiniciar a sessão
- **Autenticação** (opcional): Sistema de login para controlar acesso

### Configuração da Interface

As configurações da interface são feitas através de variáveis de ambiente no arquivo `.env`:

```bash
# Configurações da Interface Streamlit
APP_VERSION=1.0.0                    # Versão da aplicação
BACKEND_URL=http://localhost:8000    # URL do backend
BACKEND_PORT=8000                    # Porta do backend
API_URL=                             # URL da API de autenticação (opcional)
AUTH_TOKEN=                          # Token de autenticação (opcional)
POD_ID=                              # ID do POD para RunPod (opcional)
```

### Monitoramento

A interface registra automaticamente:
- **Histórico de perguntas**: `monitoramento/history.log`
- **Erros**: `monitoramento/erros.log`
- **Feedback dos usuários**: `monitoramento/feedback.log`

Esses logs incluem:
- Timestamp
- Sessão ID
- Pergunta e resposta
- Latência
- Modo de operação (RAG, direto, etc.)
- Informações de uso

## 🏗️ Arquitetura

```
ubchat-agentic-rag/
├── main.py                    # API Flask (Backend)
├── ui_app.py                  # Interface Streamlit (Frontend)
├── config.py                  # Configurações centralizadas
├── requirements.txt           # Dependências
├── .env.example              # Template de variáveis de ambiente
├── agents/
│   ├── __init__.py
│   ├── query_router.py       # Roteamento de consultas
│   └── rag_engine.py         # Motor RAG (Ollama + Pinecone)
├── monitoramento/            # Logs e monitoramento (criado automaticamente)
│   ├── history.log          # Histórico de interações
│   ├── erros.log            # Log de erros
│   └── feedback.log         # Feedback dos usuários
└── README.md
```

## 🔧 Configuração Avançada

### Reranking

Ative reranking para melhorar relevância:

```bash
# .env
RERANK_METHOD_DEFAULT=cross-encoder
RERANK_TOP_K_DEFAULT=3
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Namespaces

Use namespaces para isolar documentos por contexto:

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "pergunta",
    "namespace": "politicas-rh"
  }'
```

### Modelos Alternativos

Troque os modelos no `.env`:

```bash
# Para respostas mais rápidas (menor qualidade)
GENERATION_MODEL=llama3.2:1b

# Para melhor qualidade (mais lento)
GENERATION_MODEL=llama3.1:70b

# Embeddings alternativos
EMBEDDING_MODEL=nomic-embed-text:latest
```

**IMPORTANTE**: Ajuste as dimensões do índice Pinecone de acordo com o modelo de embedding escolhido.

## 🧪 Testes

```bash
# Testar roteamento
python test_query_router.py

# Testar API
python teste_api.py
```

## 📊 Monitoramento

### Logs

Configure nível de log no `.env`:

```bash
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Métricas

O sistema loga automaticamente:
- Tempo de recuperação
- Número de documentos recuperados
- Scores de relevância
- Erros e fallbacks

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT.

## 🆘 Troubleshooting

### Ollama não conecta

```bash
# Verifique se o serviço está rodando
ollama list

# Reinicie o Ollama
ollama serve
```

### Pinecone timeout

- Verifique sua API key
- Confirme que o índice existe
- Verifique conectividade com a internet

### Embeddings com dimensão errada

Certifique-se de que as dimensões do índice Pinecone correspondem ao modelo:
- `mxbai-embed-large`: 1024 dimensões
- `nomic-embed-text`: 768 dimensões
- `all-MiniLM-L6-v2`: 384 dimensões

### Modelo não encontrado

```bash
# Liste modelos instalados
ollama list

# Baixe o modelo necessário
ollama pull llama3.2:latest
```

## 📚 Documentação Adicional

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Query Router README](QUERY_ROUTER_README.md)