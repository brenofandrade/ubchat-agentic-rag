# Correções do Query Router

## Problema Reportado

O roteador não estava funcionando corretamente - todas as perguntas sempre seguiam para a mesma rota, provavelmente sempre usando RAG mesmo quando não era necessário.

## Causas Identificadas

### 1. **Falta de Logging**
- Não havia logs para identificar qual rota estava sendo escolhida
- Impossível debugar o comportamento do roteador
- Erros silenciosos não eram reportados

### 2. **Problema com `format="json"` no Ollama**
- O parâmetro `format="json"` pode não ser suportado por todos os modelos Ollama
- Quando falhava, caía silenciosamente para o fallback
- O parsing de JSON era frágil

### 3. **Regras Heurísticas Fracas**
- O fallback baseado em regras tinha keywords muito genéricas
- Não diferenciava bem entre RAG, DIRECT e CLARIFY
- Default era DIRECT, mas muitas perguntas organizacionais não eram detectadas

### 4. **Prompt do Sistema Vago**
- O prompt não deixava claro quando usar cada rota
- Faltavam exemplos específicos
- Instruções eram ambíguas

## Correções Implementadas

### ✅ 1. Logging Detalhado

Adicionado logging em todos os pontos críticos:

```python
logger.info(f"Roteando pergunta: '{question[:100]}...'")
logger.info(f"Decisão de roteamento: {decision.route.value} (confiança: {decision.confidence})")
logger.debug(f"Raciocínio: {decision.reasoning}")
logger.error(f"Erro durante roteamento LLM: {e}", exc_info=True)
```

**Benefício:** Agora é possível ver exatamente qual rota está sendo escolhida e por quê.

### ✅ 2. Remoção do `format="json"` + Parsing Robusto

**Antes:**
```python
self.client = ChatOllama(
    model=self.model,
    base_url=self.base_url,
    temperature=0.3,
    format="json"  # Problemático
)
```

**Depois:**
```python
self.client = ChatOllama(
    model=self.model,
    base_url=self.base_url,
    temperature=0.1  # Mais determinístico
)
```

Melhorado o parsing de JSON com validação:
```python
# Valida campos obrigatórios
if "route" not in result or "confidence" not in result or "reasoning" not in result:
    raise ValueError(f"JSON inválido: faltam campos obrigatórios")
```

**Benefício:** Maior compatibilidade com diferentes modelos Ollama e melhor tratamento de erros.

### ✅ 3. Regras Heurísticas Melhoradas

Expandidas as keywords e adicionada lógica em camadas:

**RAG Keywords (políticas/procedimentos da empresa):**
- política, procedimento, benefício, reembolso, férias
- home office, RH, empresa, interno
- "como solicito", "qual o processo", manual

**DIRECT Keywords (conhecimento geral):**
- "o que é", "como funciona", "explique"
- capital, história, ciência, matemática, física
- fotossíntese, definição

**CLARIFY Patterns (perguntas vagas):**
- "como faço" (sem contexto), "me ajuda"
- Perguntas < 8 caracteres
- Saudações genéricas

**Lógica em Camadas:**
1. Perguntas muito curtas → CLARIFY
2. Padrões vagos → CLARIFY
3. Keywords RAG fortes → RAG (85% confiança)
4. Keywords DIRECT → DIRECT (80% confiança)
5. Contexto organizacional → RAG (70% confiança)
6. Padrão → DIRECT (60% confiança)

**Benefício:** Muito melhor diferenciação entre os tipos de pergunta.

### ✅ 4. Prompt do Sistema Melhorado

**Mudanças principais:**
- Exemplos específicos para cada rota
- Instruções claras com checkmarks (✓)
- Ênfase em preferir RAG/DIRECT ao invés de CLARIFY
- Formato JSON explícito

**Exemplo de instrução RAG:**
```
1. **RAG** - Use quando a pergunta precisa de DOCUMENTOS INTERNOS da organização:
   ✓ Políticas da empresa (férias, benefícios, RH, etc.)
   ✓ Procedimentos internos (reembolso, aprovações, processos)

   Exemplos RAG:
   - "Qual é a política de férias da empresa?"
   - "Como solicito reembolso de despesas?"
```

**Benefício:** LLM entende muito melhor quando usar cada rota.

## Resultados Esperados

Com essas correções, o roteador agora deve:

1. ✅ **Identificar corretamente** perguntas sobre políticas/procedimentos → RAG
2. ✅ **Identificar corretamente** perguntas de conhecimento geral → DIRECT
3. ✅ **Usar CLARIFY** apenas para perguntas realmente vagas
4. ✅ **Logar todas as decisões** para facilitar debugging
5. ✅ **Falhar graciosamente** com fallback robusto se o LLM não funcionar

## Sobre o Checkbox "Usar Documentos Internos"

### Por que o checkbox não é mais necessário?

O **Query Router** agora decide automaticamente se deve buscar documentos internos (RAG) ou responder diretamente (DIRECT). Exemplos:

| Pergunta | Rota Automática | Motivo |
|----------|----------------|--------|
| "Qual a política de férias?" | **RAG** | Política da empresa |
| "Como funciona fotossíntese?" | **DIRECT** | Conhecimento geral |
| "Quantos dias de férias eu tenho?" | **RAG** | Benefícios da empresa |
| "O que é Python?" | **DIRECT** | Definição geral |

### Recomendação

**Remover o checkbox da interface** e deixar o roteador decidir automaticamente. Isso:
- ✅ Simplifica a UX (menos decisões para o usuário)
- ✅ Usa a inteligência do sistema
- ✅ Reduz erros do usuário (escolher a opção errada)

### Se quiser manter o checkbox

Se for necessário manter controle manual, considere:
- Torná-lo **opcional/avançado** (oculto por padrão)
- Usar como **override** do roteador (force RAG ou DIRECT)
- Adicionar tooltip explicando quando usar cada opção

## Testando as Correções

Para testar se o roteador está funcionando:

1. **Habilite logs detalhados** no `.env`:
```bash
LOG_LEVEL=DEBUG
```

2. **Teste perguntas variadas:**
```bash
# Deve ser RAG
POST /chat {"question": "Qual a política de férias da empresa?"}

# Deve ser DIRECT
POST /chat {"question": "Como funciona fotossíntese?"}

# Deve ser CLARIFY
POST /chat {"question": "Como faço?"}
```

3. **Verifique os logs** para ver a decisão:
```
INFO - Roteando pergunta: 'Qual a política de férias...'
INFO - Decisão de roteamento: rag (confiança: 0.95)
```

## Próximos Passos

- [ ] Testar o roteador com perguntas reais
- [ ] Ajustar keywords se necessário baseado no comportamento
- [ ] Considerar remover o checkbox da interface
- [ ] Monitorar logs de produção para identificar padrões de erro
- [ ] Possível adição de métricas de acurácia do roteador

## Arquivos Modificados

- ✏️ `agents/query_router.py` - Todas as correções implementadas
- 📄 `ROUTER_FIXES.md` - Esta documentação
