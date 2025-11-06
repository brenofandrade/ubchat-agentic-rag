"""
Script para testar o comportamento do QueryRouter
"""
import os
import sys

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import QueryRouter
from config import GENERATION_MODEL, OLLAMA_BASE_URL

# Inicializar o roteador
router = QueryRouter(
    model=os.getenv("LLM_MODEL", GENERATION_MODEL),
    provider=os.getenv("LLM_PROVIDER", "ollama"),
    base_url=OLLAMA_BASE_URL
)

# Perguntas de teste
test_questions = [
    "Qual é a política de férias da empresa?",  # Deveria ser RAG
    "Como funciona fotossíntese?",  # Deveria ser DIRECT
    "Como faço?",  # Deveria ser CLARIFY
    "Quantos dias de férias eu tenho direito?",  # Deveria ser RAG
    "O que é Python?",  # Deveria ser DIRECT
    "Qual a capital da França?",  # Deveria ser DIRECT
    "Como solicito reembolso de despesas?",  # Deveria ser RAG
]

print("=" * 80)
print("TESTANDO COMPORTAMENTO DO QUERY ROUTER")
print("=" * 80)
print(f"Provider: {router.provider}")
print(f"Model: {router.model}")
print(f"Client: {router.client}")
print("=" * 80)
print()

for question in test_questions:
    print(f"Pergunta: {question}")
    decision = router.route_query(question)
    print(f"  → Rota: {decision.route.value}")
    print(f"  → Confiança: {decision.confidence}")
    print(f"  → Raciocínio: {decision.reasoning}")
    if decision.clarifying_questions:
        print(f"  → Perguntas: {decision.clarifying_questions}")
    if decision.suggested_documents:
        print(f"  → Documentos: {decision.suggested_documents}")
    print()
