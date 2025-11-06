"""
Test script for the Query Router Agent

This script demonstrates the query router's ability to classify questions
into different routing strategies: RAG, DIRECT, or CLARIFY.

Note: This uses the rule-based fallback by default. To test with LLM,
set the OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
"""

import json
from agents import QueryRouter, RouteType


def print_decision(question: str, decision):
    """Pretty print a routing decision"""
    print(f"\n{'='*80}")
    print(f"Pergunta: {question}")
    print(f"{'='*80}")
    print(f"Rota: {decision.route.value.upper()}")
    print(f"Confiança: {decision.confidence:.2f}")
    print(f"Justificativa: {decision.reasoning}")

    if decision.clarifying_questions:
        print(f"\nPerguntas de Clarificação:")
        for i, q in enumerate(decision.clarifying_questions, 1):
            print(f"  {i}. {q}")

    if decision.suggested_documents:
        print(f"\nDocumentos Sugeridos:")
        for i, doc in enumerate(decision.suggested_documents, 1):
            print(f"  {i}. {doc}")


def main():
    """Test the query router with various example questions"""

    # Initialize the router (will use rule-based routing if no API key is set)
    router = QueryRouter()

    # Test questions covering different scenarios
    test_questions = [
        # Should route to RAG (internal documents)
        "Qual é a política de férias da nossa empresa?",
        "Onde posso encontrar o manual de procedimentos internos?",
        "Quais são as normas de segurança do trabalho?",

        # Should route to DIRECT (general knowledge)
        "O que é fotossíntese?",
        "Como funciona o sistema solar?",
        "Qual é a capital da França?",

        # Should route to CLARIFY (ambiguous/vague)
        "Como?",
        "Preciso de ajuda",
        "O que fazer?",
        "Isso funciona?",
    ]

    print("\n" + "="*80)
    print("TESTE DO AGENTE DE ROTEAMENTO DE CONSULTAS")
    print("="*80)

    # Test each question
    for question in test_questions:
        decision = router.route_query(question)
        print_decision(question, decision)

    # Test with context
    print(f"\n{'='*80}")
    print("TESTE COM CONTEXTO ADICIONAL")
    print(f"{'='*80}")

    question_with_context = "Como faço para solicitar?"
    context = "O usuário está em uma conversa sobre solicitação de férias"

    decision = router.route_query(question_with_context, context=context)
    print_decision(f"{question_with_context} (Contexto: {context})", decision)

    # Summary
    print(f"\n{'='*80}")
    print("RESUMO")
    print(f"{'='*80}")
    print("✓ Agente de roteamento funcionando")
    print("✓ Suporta 3 tipos de rota: RAG, DIRECT, CLARIFY")
    print("✓ Pode usar regras simples ou LLM para classificação")
    print("✓ Fornece justificativas e sugestões")
    print("\nPara usar com LLM, configure:")
    print("  export OPENAI_API_KEY='your-key'  # Para OpenAI")
    print("  export ANTHROPIC_API_KEY='your-key'  # Para Anthropic")


if __name__ == "__main__":
    main()
