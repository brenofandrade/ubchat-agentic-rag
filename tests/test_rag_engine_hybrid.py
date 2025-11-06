import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _reload_module(monkeypatch, enable_hybrid: str) -> object:
    monkeypatch.setenv("PINECONE_API_KEY_DSUNIBLU", "test-key")
    monkeypatch.setenv("PINECONE_INDEX", "test-index")
    monkeypatch.setenv("ENABLE_HYBRID_SEARCH", enable_hybrid)
    monkeypatch.setenv("HYBRID_ALPHA", "0.6")
    monkeypatch.setenv("SPLADE_MODEL", "test-splade")

    for module_name in ["agents.rag_engine", "config"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    module = importlib.import_module("agents.rag_engine")
    return importlib.reload(module)


@pytest.fixture
def rag_engine_module(monkeypatch):
    return _reload_module(monkeypatch, "true")


@pytest.fixture
def rag_engine_module_dense(monkeypatch):
    return _reload_module(monkeypatch, "false")


def _setup_common_stubs(monkeypatch, module):
    class DummyIndex:
        def __init__(self):
            self.queries = []

        def describe_index_stats(self, namespace=None):
            return {"dimension": 3, "namespaces": {}}

        def query(self, **kwargs):
            self.queries.append(kwargs)
            return SimpleNamespace(matches=[
                SimpleNamespace(
                    metadata={"text": "Documento híbrido", "source": "unit-test"},
                    score=0.91,
                )
            ])

    dummy_index = DummyIndex()

    class DummyPinecone:
        def __init__(self, api_key):
            self.api_key = api_key

        def Index(self, name):
            return dummy_index

    monkeypatch.setattr(module, "Pinecone", DummyPinecone)

    class DummyChat:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, prompt):
            return SimpleNamespace(content="Resposta gerada")

    monkeypatch.setattr(module, "ChatOllama", DummyChat)

    class DummyEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(module, "OllamaEmbeddings", DummyEmbeddings)

    vectorstore = SimpleNamespace(similarity_search_with_score=lambda **kwargs: [])
    monkeypatch.setattr(module, "PineconeVectorStore", lambda **_: vectorstore)

    return dummy_index, vectorstore


def test_hybrid_retrieve_combines_dense_and_sparse(monkeypatch, rag_engine_module):
    dummy_index, _ = _setup_common_stubs(monkeypatch, rag_engine_module)

    class DummySparseEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode_queries(self, queries):
            return [{"indices": [1, 2], "values": [0.4, 0.2]}]

    def dummy_scale(dense, sparse, alpha):
        return [value * alpha for value in dense], {
            "indices": sparse["indices"],
            "values": [value * (1 - alpha) for value in sparse["values"]],
        }

    monkeypatch.setattr(rag_engine_module, "SpladeSparseEncoder", DummySparseEncoder)
    monkeypatch.setattr(rag_engine_module, "hybrid_convex_scale", dummy_scale)

    engine = rag_engine_module.RAGEngine()

    documents = engine.retrieve("teste híbrido", top_k=1)

    assert len(documents) == 1
    assert documents[0].content == "Documento híbrido"
    assert dummy_index.queries, "A consulta híbrida deve chamar Pinecone.query"

    query_kwargs = dummy_index.queries[0]
    assert "sparse_vector" in query_kwargs
    assert query_kwargs["namespace"] == rag_engine_module.DEFAULT_NAMESPACE


def test_dense_retrieve_used_when_hybrid_disabled(monkeypatch, rag_engine_module_dense):
    _, vectorstore = _setup_common_stubs(monkeypatch, rag_engine_module_dense)

    dense_document = SimpleNamespace(
        page_content="Documento denso",
        metadata={"source": "unit-test"},
    )

    vectorstore.similarity_search_with_score = lambda **kwargs: [
        (dense_document, 0.55)
    ]

    engine = rag_engine_module_dense.RAGEngine()

    documents = engine.retrieve("teste denso", top_k=1)

    assert len(documents) == 1
    assert documents[0].content == "Documento denso"


def test_retrieve_with_variations_when_no_results(monkeypatch, rag_engine_module):
    """Testa que quando não encontra documentos, tenta buscar com variações"""
    dummy_index, vectorstore = _setup_common_stubs(monkeypatch, rag_engine_module)

    # Simula nenhum resultado na busca inicial
    call_count = {"count": 0}

    def mock_similarity_search(**kwargs):
        call_count["count"] += 1
        # Primeira chamada retorna vazio, segunda retorna documento
        if call_count["count"] == 1:
            return []
        return [
            (
                SimpleNamespace(
                    page_content="Documento encontrado com variação",
                    metadata={"source": "test"}
                ),
                0.75
            )
        ]

    vectorstore.similarity_search_with_score = mock_similarity_search

    # Mock para geração de variações
    class DummyChat:
        def invoke(self, prompt):
            return SimpleNamespace(
                content="termo alternativo\noutra busca\nvariação da consulta"
            )

    monkeypatch.setattr(rag_engine_module, "ChatOllama", DummyChat)

    engine = rag_engine_module.RAGEngine()
    engine.llm = DummyChat()

    # Desabilita hybrid para simplificar o teste
    engine.use_hybrid = False
    engine.sparse_encoder = None

    documents = engine.retrieve("busca sem resultados", top_k=3, enable_fallback=True)

    # Deve ter encontrado documentos com as variações
    assert len(documents) > 0
    assert documents[0].content == "Documento encontrado com variação"
    assert call_count["count"] > 1, "Deve ter tentado múltiplas buscas"


def test_generate_simple_variations(monkeypatch, rag_engine_module):
    """Testa geração de variações simples com regras"""
    _, _ = _setup_common_stubs(monkeypatch, rag_engine_module)

    engine = rag_engine_module.RAGEngine()

    # Testa variação com sinônimo
    variations = engine._generate_simple_variations("Como solicitar férias")
    assert len(variations) > 0
    # Deve conter alguma variação com sinônimo
    assert any("pedir" in v or "recesso" in v for v in variations)


def test_query_returns_suggestions_when_no_documents(monkeypatch, rag_engine_module):
    """Testa que quando não encontra documentos, retorna sugestões ao usuário"""
    _, vectorstore = _setup_common_stubs(monkeypatch, rag_engine_module)

    # Simula nenhum resultado mesmo com variações
    vectorstore.similarity_search_with_score = lambda **kwargs: []

    class DummyChat:
        def invoke(self, prompt):
            if "variações" in prompt.lower() or "gere" in prompt.lower():
                return SimpleNamespace(content="termo alternativo\noutra busca")
            return SimpleNamespace(content="sugestão 1\nsugestão 2\nsugestão 3")

    monkeypatch.setattr(rag_engine_module, "ChatOllama", DummyChat)

    engine = rag_engine_module.RAGEngine()
    engine.llm = DummyChat()
    engine.use_hybrid = False
    engine.sparse_encoder = None

    result = engine.query("busca impossível")

    # Verifica que retorna mensagem apropriada
    assert "não encontrei documentos relevantes" in result["answer"].lower()
    assert "você poderia tentar buscar por" in result["answer"].lower() or "fornecer mais detalhes" in result["answer"].lower()
    assert result["metadata"]["retrieved_count"] == 0
    assert "search_suggestions" in result["metadata"]
