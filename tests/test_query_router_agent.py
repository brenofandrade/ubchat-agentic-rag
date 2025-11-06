import os
import pathlib
import sys

# Ensure project root on sys.path before importing project modules
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure required configuration for config validation
os.environ.setdefault("PINECONE_API_KEY_DSUNIBLU", "test-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")

from agents import QueryRouter, RouteDecision, RouteType  # noqa: E402


def _build_router() -> QueryRouter:
    router = QueryRouter(provider="ollama", model="llama3.2:latest")
    # Force fallback heuristics to avoid external service dependency
    router.client = None
    return router


def test_routes_internal_policy_question_to_rag():
    router = _build_router()

    decision = router.route_query("Qual é a política de férias da empresa?")

    assert isinstance(decision, RouteDecision)
    assert decision.route is RouteType.RAG
    assert decision.suggested_documents is not None
    assert "company_policies" in decision.suggested_documents


def test_routes_general_knowledge_question_to_direct():
    router = _build_router()

    decision = router.route_query("O que é fotossíntese?")

    assert decision.route is RouteType.DIRECT
    assert decision.reasoning
    assert decision.confidence >= 0.6


def test_routes_vague_question_to_clarify_with_follow_ups():
    router = _build_router()

    decision = router.route_query("Preciso de ajuda")

    assert decision.route is RouteType.CLARIFY
    assert decision.clarifying_questions
    assert any("específico" in q.lower() for q in decision.clarifying_questions)
