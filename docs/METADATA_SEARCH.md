# Busca por Metadados

## Visão Geral

O sistema RAG agora suporta **busca automática por metadados** quando o usuário mencionar identificadores específicos de documentos na consulta. Esta funcionalidade permite recuperar documentos específicos de forma mais precisa quando o usuário conhece o código ou identificador do documento.

## Como Funciona

### Detecção Automática de Identificadores

O sistema detecta automaticamente padrões comuns de identificadores de documentos, incluindo:

- **MAN-XXX**: Manuais (ex: MAN-297)
- **NR-XXX**: Normas Regulamentadoras (ex: NR-013, NR-12)
- **ISO-XXX**: Normas ISO (ex: ISO-9001)
- **PROC-XXX**: Procedimentos (ex: PROC-1234)
- **Outros padrões**: Qualquer código no formato `LETRAS-NÚMEROS` ou `LETRAS NÚMEROS`

### Exemplos de Uso

#### Exemplo 1: Manual Específico
```
Usuário: "O que diz o manual MAN-297?"

Sistema:
1. Detecta o identificador: MAN-297
2. Cria filtros de metadados para buscar documentos com esse código
3. Retorna documentos que correspondem ao MAN-297
```

#### Exemplo 2: Norma Regulamentadora
```
Usuário: "Como se adequar à NR-013?"

Sistema:
1. Detecta o identificador: NR-013
2. Normaliza para formato padrão (NR-013)
3. Busca documentos relacionados à NR-013
```

#### Exemplo 3: Múltiplos Documentos
```
Usuário: "Compare o manual MAN-297 com a norma NR-013"

Sistema:
1. Detecta múltiplos identificadores: MAN-297, NR-013
2. Cria filtros para buscar ambos os documentos
3. Retorna documentos de ambos os códigos
```

#### Exemplo 4: Consulta Genérica (Sem Identificadores)
```
Usuário: "Como fazer manutenção preventiva?"

Sistema:
1. Não detecta identificadores específicos
2. Usa busca semântica normal
3. Retorna documentos relevantes baseados em similaridade
```

## Características Técnicas

### Normalização de Identificadores

O sistema normaliza automaticamente os identificadores para um formato padrão:

- **Case insensitive**: `man-297` → `MAN-297`
- **Espaços**: `MAN 297` → `MAN-297`
- **Hífens**: `NR13` → `NR-13`

### Campos de Metadados Pesquisados

O sistema busca o identificador nos seguintes campos de metadados do Pinecone:

- `document_id`
- `doc_id`
- `id`
- `source`
- `title`
- `name`

Isso garante compatibilidade com diferentes estruturas de metadados.

### Filtros do Pinecone

O sistema utiliza a sintaxe de filtros do Pinecone (estilo MongoDB) para criar condições de busca:

```python
# Para um único identificador
{
  "$or": [
    {"document_id": {"$eq": "MAN-297"}},
    {"source": {"$eq": "MAN-297"}},
    # ... outros campos
  ]
}

# Para múltiplos identificadores
{
  "$or": [
    {"document_id": {"$eq": "MAN-297"}},
    {"document_id": {"$eq": "NR-013"}},
    # ... todos os campos para ambos identificadores
  ]
}
```

## Integração com o Sistema

### No Código Python

```python
from agents.rag_engine import RAGEngine

# Criar instância do RAG Engine
engine = RAGEngine(namespace="seu-namespace")

# Busca automática por metadados (padrão)
documents = engine.retrieve("O que diz o manual MAN-297?")
# O sistema detecta automaticamente o identificador e aplica filtros

# Desabilitar detecção automática
documents = engine.retrieve(
    "O que diz o manual MAN-297?",
    auto_detect_identifiers=False
)

# Fornecer filtros personalizados
custom_filters = {
    "source": {"$eq": "manual-especifico.pdf"}
}
documents = engine.retrieve(
    "Busca qualquer coisa",
    metadata_filters=custom_filters
)
```

### Via API REST

A funcionalidade está disponível automaticamente em todos os endpoints de busca:

```bash
# POST /rag/query
curl -X POST http://localhost:5000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "O que diz o manual MAN-297?",
    "chat_history": []
  }'

# POST /rag/retrieve
curl -X POST http://localhost:5000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Como se adequar à NR-013?",
    "top_k": 5
  }'
```

## Logs e Debugging

Quando identificadores são detectados, o sistema registra no log:

```
INFO - 🔍 Identificadores detectados: ['MAN-297'] - aplicando busca por metadados
INFO - Recuperando 5 documentos com filtros de metadados para query: 'O que diz...'
INFO - ✓ Recuperados 3 documentos (busca híbrida com filtros de metadados)
```

## Configuração de Metadados no Pinecone

Para aproveitar ao máximo esta funcionalidade, certifique-se de que seus documentos no Pinecone incluam metadados estruturados:

```python
# Exemplo de estrutura de metadados recomendada
metadata = {
    "document_id": "MAN-297",     # Identificador único do documento
    "title": "Manual de Operação 297",
    "source": "manual-297.pdf",
    "doc_type": "manual",
    "page": 1,
    "section": "Introdução"
}
```

## Padrões de Identificação

### Padrões Suportados

| Padrão | Regex | Exemplo |
|--------|-------|---------|
| Geral | `[A-Z]{2,6}-\d{2,6}` | MAN-297, ISO-9001, PROC-1234 |
| Com espaço | `[A-Z]{2,6}\s*\d{2,6}` | MAN 297, ISO 9001 |
| NR específico | `NR\s*-?\s*\d{1,3}` | NR-13, NR 12, NR013 |

### Adicionando Novos Padrões

Para adicionar suporte a novos padrões de identificadores, edite a função `extract_document_identifiers()` em `agents/rag_engine.py`:

```python
def extract_document_identifiers(query: str) -> List[str]:
    patterns = [
        r'\b([A-Z]{2,6}-\d{2,6})\b',
        r'\b([A-Z]{2,6}\s*\d{2,6})\b',
        r'\b(NR\s*-?\s*\d{1,3})\b',
        # Adicione seu padrão customizado aqui:
        r'\b(SEU-PADRAO-\d+)\b',
    ]
    # ...
```

## Benefícios

1. **Precisão**: Retorna exatamente o documento solicitado quando identificadores são fornecidos
2. **Velocidade**: Filtros de metadados são mais rápidos que busca semântica completa
3. **Transparência**: Logs claros indicam quando busca por metadados é aplicada
4. **Flexibilidade**: Funciona automaticamente ou pode ser controlado manualmente
5. **Compatibilidade**: Funciona com busca densa e híbrida

## Limitações e Considerações

1. **Qualidade dos Metadados**: A eficácia depende da qualidade dos metadados no Pinecone
2. **Falsos Positivos**: Códigos que parecem identificadores mas não são podem ser detectados
3. **Fallback**: Se nenhum documento for encontrado com filtros, não há fallback automático para busca semântica
4. **Campos Personalizados**: Pode ser necessário ajustar os campos pesquisados para seu caso de uso

## Testando

Execute os testes para validar a funcionalidade:

```bash
# Teste isolado da lógica
python test_metadata_logic.py

# Testes com pytest (se disponível)
pytest tests/test_metadata_search.py -v
```

## Próximos Passos

Possíveis melhorias futuras:

1. Suporte a ranges de documentos (ex: "MAN-297 até MAN-300")
2. Busca fuzzy para identificadores similares
3. Sugestão de documentos relacionados
4. Cache de identificadores frequentes
5. API para registrar novos padrões de identificadores dinamicamente
