# Query Router Agent 🤖

Um agente inteligente que analisa perguntas e decide a melhor estratégia para respondê-las.

## 🎯 Funcionalidades

O Query Router classifica perguntas em três categorias:

### 1. **RAG** (Retrieval-Augmented Generation)
- Consulta documentos internos
- Usado quando a pergunta requer informações específicas da organização
- Exemplo: *"Qual é a política de férias da empresa?"*

### 2. **DIRECT** (Resposta Direta)
- Responde com conhecimento do próprio modelo
- Usado para perguntas de conhecimento geral
- Exemplo: *"Como funciona fotossíntese?"*

### 3. **CLARIFY** (Clarificação)
- Solicita mais informações ao usuário
- Usado quando a pergunta é vaga ou ambígua
- Exemplo: *"Como faço isso?"*

## 🚀 Como Usar

### 1. Instalação

```bash
pip install -r requirements.txt
```

### 2. Configuração (Opcional)

Para usar LLM ao invés de regras simples:

```bash
# OpenAI
export OPENAI_API_KEY='sua-chave-aqui'
export LLM_PROVIDER='openai'
export LLM_MODEL='gpt-4'

# Anthropic Claude
export ANTHROPIC_API_KEY='sua-chave-aqui'
export LLM_PROVIDER='anthropic'
export LLM_MODEL='claude-3-opus-20240229'
```

### 3. Uso Programático

```python
from agents import QueryRouter

# Inicializar o router
router = QueryRouter()

# Rotear uma pergunta
decision = router.route_query("Qual é a política de férias?")

print(f"Rota: {decision.route}")  # RouteType.RAG
print(f"Confiança: {decision.confidence}")  # 0.85
print(f"Justificativa: {decision.reasoning}")
```

### 4. API REST

Inicie o servidor:

```bash
python main.py
```

#### Endpoint Completo

```bash
curl -X POST http://localhost:8000/route-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual é a política de férias da empresa?",
    "context": "Conversa sobre benefícios"
  }'
```

Resposta:
```json
{
  "route": "rag",
  "confidence": 0.85,
  "reasoning": "Pergunta sobre política interna da empresa",
  "suggested_documents": ["company_policies", "hr_manual"]
}
```

#### Endpoint Simplificado

```bash
curl -X POST http://localhost:8000/route-query/simple \
  -H "Content-Type: application/json" \
  -d '{"question": "Como funciona fotossíntese?"}'
```

Resposta:
```json
{
  "route": "direct"
}
```

## 🧪 Testes

Execute o script de teste:

```bash
python test_query_router.py
```

Isso testará o router com vários exemplos de perguntas.

## 🏗️ Arquitetura

```
agents/
├── __init__.py           # Exports principais
└── query_router.py       # Implementação do agente
    ├── RouteType         # Enum com tipos de rota
    ├── RouteDecision     # Decisão estruturada
    └── QueryRouter       # Classe principal
```

## 🎨 Modos de Operação

### Modo Rule-Based (Padrão)
- Usa regras heurísticas simples
- Não requer API keys
- Bom para casos básicos
- Rápido e sem custos

### Modo LLM
- Usa modelos de linguagem para análise inteligente
- Requer API key (OpenAI ou Anthropic)
- Mais preciso e adaptável
- Melhor para casos complexos

## 📊 Exemplos de Classificação

| Pergunta | Rota | Motivo |
|----------|------|--------|
| "Qual é a política de férias?" | RAG | Informação interna |
| "O que é fotossíntese?" | DIRECT | Conhecimento geral |
| "Como?" | CLARIFY | Muito vaga |
| "Onde encontro o manual?" | RAG | Documento interno |
| "Qual a capital da França?" | DIRECT | Conhecimento geral |
| "Preciso de ajuda" | CLARIFY | Sem contexto |

## 🔧 Personalização

### Adicionar Novos Keywords (Rule-Based)

Edite `query_router.py` na função `_rule_based_routing`:

```python
rag_keywords = [
    "documento", "política", "procedimento",
    # Adicione seus keywords aqui
    "contrato", "regulamento"
]
```

### Customizar Prompt (LLM)

Edite `query_router.py` na função `_llm_based_routing`:

```python
system_prompt = """
Você é um agente de roteamento...
[Adicione suas instruções customizadas aqui]
"""
```

## 🔍 Debugging

Para ver os logs de decisão:

```python
decision = router.route_query(question)
print(json.dumps({
    "route": decision.route.value,
    "confidence": decision.confidence,
    "reasoning": decision.reasoning
}, indent=2))
```

## 📈 Próximos Passos

Ideias para expandir o agente:

1. **Feedback Loop**: Aprender com decisões corretas/incorretas
2. **Multi-RAG**: Diferentes fontes de documentos
3. **Hybrid Routing**: Combinar múltiplas estratégias
4. **Analytics**: Dashboards de métricas de roteamento
5. **A/B Testing**: Comparar estratégias de roteamento

## 🤝 Contribuindo

Para adicionar novos tipos de rota:

1. Adicione ao enum `RouteType`
2. Atualize a lógica de `_rule_based_routing`
3. Atualize o prompt em `_llm_based_routing`
4. Adicione testes em `test_query_router.py`

## 📝 License

Este projeto faz parte do sistema Agentic RAG.
