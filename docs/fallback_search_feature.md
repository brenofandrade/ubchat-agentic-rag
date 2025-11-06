# Funcionalidade de Busca com Fallback

## Descrição

Quando o sistema não consegue encontrar documentos relevantes para uma consulta, agora ele automaticamente:

1. **Tenta buscar com termos semelhantes**: Gera variações da consulta original usando sinônimos e termos relacionados
2. **Sugere alternativas ao usuário**: Se ainda não encontrar nada, retorna sugestões de termos alternativos e pede mais detalhes

## Como Funciona

### 1. Busca com Variações

Quando uma busca inicial não retorna resultados, o sistema:

- Gera 3 variações da consulta original usando:
  - OpenAI GPT-3.5 (se disponível)
  - Ollama (fallback)
  - Regras simples baseadas em sinônimos (fallback final)

- Tenta buscar com cada variação até encontrar documentos

**Exemplo:**
```
Consulta original: "Como solicitar férias"
Variações geradas:
  1. "Como pedir recesso"
  2. "Procedimento para requisitar período de descanso"
  3. "Fazer pedido férias"
```

### 2. Sugestões ao Usuário

Se nenhum documento for encontrado mesmo com as variações, o sistema retorna uma resposta estruturada com:

- Mensagem informando que não encontrou resultados
- Lista de termos alternativos sugeridos
- Orientações para o usuário fornecer mais detalhes

**Exemplo de resposta:**
```
Desculpe, não encontrei documentos relevantes para sua pergunta.

Você poderia tentar buscar por:
1. Política de férias
2. Processo de solicitação de recesso
3. Benefícios de tempo livre

Ou você pode fornecer mais detalhes sobre:
- O contexto da sua pergunta
- Termos específicos relacionados ao que você procura
- Uma reformulação da sua pergunta com mais informações
```

## API

### RAGEngine.retrieve()

```python
def retrieve(
    self,
    query: str,
    top_k: Optional[int] = None,
    enable_fallback: bool = True
) -> List[RetrievedDocument]:
    """
    Recupera documentos relevantes do vector store.

    Args:
        query: Consulta do usuário
        top_k: Número de documentos a recuperar
        enable_fallback: Se True, tenta buscar com termos alternativos

    Returns:
        Lista de documentos recuperados
    """
```

### Métodos Auxiliares

#### `_generate_query_variations(query, num_variations=3)`
Gera variações da consulta usando LLM ou regras

#### `_retrieve_with_variations(query, k)`
Tenta recuperar documentos com cada variação gerada

#### `_generate_search_suggestions(query)`
Gera sugestões de termos alternativos para o usuário

#### `_generate_simple_variations(query)`
Fallback baseado em regras para gerar variações

## Configuração

A funcionalidade está habilitada por padrão. Para desabilitar:

```python
# Ao chamar retrieve diretamente
documents = rag_engine.retrieve(query, enable_fallback=False)
```

## Fluxo de Execução

```
1. Busca inicial (densa ou híbrida)
   ↓
2. Se não encontrar resultados:
   ↓
3. Gera variações da consulta
   ↓
4. Tenta buscar com cada variação
   ↓
5. Se ainda não encontrar:
   ↓
6. Retorna sugestões e pede mais detalhes
```

## Testes

Novos testes foram adicionados em `tests/test_rag_engine_hybrid.py`:

- `test_retrieve_with_variations_when_no_results`: Verifica busca com variações
- `test_generate_simple_variations`: Testa geração de variações por regras
- `test_query_returns_suggestions_when_no_documents`: Verifica sugestões ao usuário

Execute os testes com:
```bash
pytest tests/test_rag_engine_hybrid.py -v
```

## Benefícios

✅ **Maior taxa de sucesso**: Encontra documentos mesmo quando a consulta não é perfeita

✅ **Melhor UX**: Usuário recebe orientação quando não há resultados

✅ **Fallback robusto**: Funciona mesmo sem OpenAI, usando Ollama ou regras

✅ **Não invasivo**: Pode ser desabilitado facilmente se necessário
