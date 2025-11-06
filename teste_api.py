import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from typing import Optional, Dict, Any, List


def test_health_check(base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
    """
    Testa a rota GET /health

    Verifica se o servidor está funcionando e retorna informações de status.
    """
    url = f"{base_url}/health"
    print(f"\n{'='*60}")
    print(f"Testando: GET {url}")
    print(f"{'='*60}")

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")

        # Validações
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "status" in data, "Missing 'status' field"
        assert data["status"] == "ok", f"Expected status 'ok', got '{data['status']}'"

        print("✓ Teste passou!")

    except ConnectionError:
        print(f"❌ Connection error when calling {url}. Is the server running?")
    except Timeout:
        print(f"❌ Request to {url} timed out.")
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"❌ HTTP error {status} from {url}: {e}")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
    except RequestException as e:
        print(f"❌ Unexpected error calling {url}: {e}")


def test_route_query(
    question: str,
    context: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    timeout: int = 1200
) -> None:
    """
    Testa a rota POST /route-query

    Roteia uma pergunta para determinar a melhor estratégia (rag, direct, clarify).
    """
    url = f"{base_url}/route-query"
    print(f"\n{'='*60}")
    print(f"Testando: POST {url}")
    print(f"{'='*60}")

    payload: Dict[str, Any] = {"question": question}
    if context:
        payload["context"] = context

    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")

        # Validações
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "route" in data, "Missing 'route' field"
        assert data["route"] in ["rag", "direct", "clarify"], f"Invalid route: {data['route']}"
        assert "confidence" in data, "Missing 'confidence' field"
        assert "reasoning" in data, "Missing 'reasoning' field"

        print("✓ Teste passou!")

    except ConnectionError:
        print(f"❌ Connection error calling {url}. Is the server running?")
    except Timeout:
        print(f"❌ Request to {url} timed out.")
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"❌ HTTP error {status} from {url}: {e}")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
    except RequestException as e:
        print(f"❌ Unexpected error calling {url}: {e}")


def test_route_query_simple(
    question: str,
    base_url: str = "http://localhost:8000",
    timeout: int = 1200
) -> None:
    """
    Testa a rota POST /route-query/simple

    Versão simplificada que retorna apenas o tipo de rota.
    """
    url = f"{base_url}/route-query/simple"
    print(f"\n{'='*60}")
    print(f"Testando: POST {url}")
    print(f"{'='*60}")

    payload = {"question": question}
    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")

        # Validações
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "route" in data, "Missing 'route' field"
        assert data["route"] in ["rag", "direct", "clarify"], f"Invalid route: {data['route']}"

        print("✓ Teste passou!")

    except ConnectionError:
        print(f"❌ Connection error calling {url}. Is the server running?")
    except Timeout:
        print(f"❌ Request to {url} timed out.")
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"❌ HTTP error {status} from {url}: {e}")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
    except RequestException as e:
        print(f"❌ Unexpected error calling {url}: {e}")


def test_rag_query(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: Optional[int] = None,
    namespace: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    timeout: int = 1200
) -> None:
    """
    Testa a rota POST /rag/query

    Executa uma consulta RAG completa: recuperação + geração.
    """
    url = f"{base_url}/rag/query"
    print(f"\n{'='*60}")
    print(f"Testando: POST {url}")
    print(f"{'='*60}")

    payload: Dict[str, Any] = {"question": question}
    if chat_history is not None:
        payload["chat_history"] = chat_history
    if top_k is not None:
        payload["top_k"] = top_k
    if namespace is not None:
        payload["namespace"] = namespace

    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")

        # Validações
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "answer" in data, "Missing 'answer' field"
        assert "documents" in data, "Missing 'documents' field"
        assert "metadata" in data, "Missing 'metadata' field"
        assert isinstance(data["documents"], list), "Documents should be a list"

        print("✓ Teste passou!")

    except ConnectionError:
        print(f"❌ Connection error calling {url}. Is the server running?")
    except Timeout:
        print(f"❌ Request to {url} timed out.")
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"❌ HTTP error {status} from {url}: {e}")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
    except RequestException as e:
        print(f"❌ Unexpected error calling {url}: {e}")


def test_rag_retrieve(
    question: str,
    top_k: Optional[int] = None,
    namespace: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    timeout: int = 1200
) -> None:
    """
    Testa a rota POST /rag/retrieve

    Recupera documentos relevantes sem gerar uma resposta.
    """
    url = f"{base_url}/rag/retrieve"
    print(f"\n{'='*60}")
    print(f"Testando: POST {url}")
    print(f"{'='*60}")

    payload: Dict[str, Any] = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if namespace is not None:
        payload["namespace"] = namespace

    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")

        # Validações
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "documents" in data, "Missing 'documents' field"
        assert "count" in data, "Missing 'count' field"
        assert isinstance(data["documents"], list), "Documents should be a list"
        assert data["count"] == len(data["documents"]), "Count doesn't match documents length"

        print("✓ Teste passou!")

    except ConnectionError:
        print(f"❌ Connection error calling {url}. Is the server running?")
    except Timeout:
        print(f"❌ Request to {url} timed out.")
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"❌ HTTP error {status} from {url}: {e}")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
    except RequestException as e:
        print(f"❌ Unexpected error calling {url}: {e}")


def test_chat(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    context: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    timeout: int = 1200
) -> None:
    """
    Testa a rota POST /chat

    Endpoint completo que usa roteamento de consulta + RAG.
    """
    url = f"{base_url}/chat"
    print(f"\n{'='*60}")
    print(f"Testando: POST {url}")
    print(f"{'='*60}")

    payload: Dict[str, Any] = {"question": question}
    if chat_history is not None:
        payload["chat_history"] = chat_history
    if context is not None:
        payload["context"] = context

    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")

        # Validações
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "route" in data, "Missing 'route' field"
        assert data["route"] in ["rag", "direct", "clarify"], f"Invalid route: {data['route']}"
        assert "confidence" in data, "Missing 'confidence' field"
        assert "reasoning" in data, "Missing 'reasoning' field"

        # Validações específicas por tipo de rota
        if data["route"] == "rag":
            assert "answer" in data, "Missing 'answer' field for RAG route"
            assert "documents" in data, "Missing 'documents' field for RAG route"
        elif data["route"] == "direct":
            assert "answer" in data, "Missing 'answer' field for DIRECT route"
        elif data["route"] == "clarify":
            assert "clarifying_questions" in data, "Missing 'clarifying_questions' for CLARIFY route"

        print("✓ Teste passou!")

    except ConnectionError:
        print(f"❌ Connection error calling {url}. Is the server running?")
    except Timeout:
        print(f"❌ Request to {url} timed out.")
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"❌ HTTP error {status} from {url}: {e}")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
    except RequestException as e:
        print(f"❌ Unexpected error calling {url}: {e}")


def test_missing_question_field(
    base_url: str = "http://localhost:8000",
    timeout: int = 30
) -> None:
    """
    Testa se as rotas retornam erro 400 quando 'question' está ausente.
    """
    endpoints = ["/route-query", "/route-query/simple", "/rag/query", "/rag/retrieve", "/chat"]

    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"\n{'='*60}")
        print(f"Testando validação: POST {url} (sem 'question')")
        print(f"{'='*60}")

        try:
            response = requests.post(url, json={}, timeout=timeout)

            print(f"Status Code: {response.status_code}")
            data = response.json()
            print(f"Response: {data}")

            # Validações
            assert response.status_code == 400, f"Expected 400, got {response.status_code}"
            assert "error" in data, "Missing 'error' field"

            print("✓ Teste de validação passou!")

        except ConnectionError:
            print(f"❌ Connection error calling {url}. Is the server running?")
        except Timeout:
            print(f"❌ Request to {url} timed out.")
        except AssertionError as e:
            print(f"❌ Assertion failed: {e}")
        except RequestException as e:
            print(f"❌ Unexpected error calling {url}: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INICIANDO TESTES DA API")
    print("="*60)

    # 1. Teste de Health Check
    test_health_check()

    # 2. Teste de Route Query (completo)
    test_route_query(
        question="O que é a Universidade Federal de Uberlândia?",
        context="Preciso de informações sobre a UFU"
    )

    # 3. Teste de Route Query (simples)
    test_route_query_simple(
        question="Qual é a capital do Brasil?"
    )

    # 4. Teste de RAG Query
    test_rag_query(
        question="Quais são os cursos oferecidos pela UFU?",
        top_k=3
    )

    # 5. Teste de RAG Retrieve
    test_rag_retrieve(
        question="Onde fica a UFU?",
        top_k=5
    )

    # 6. Teste de Chat
    test_chat(
        question="Me fale sobre a história da UFU",
        chat_history=[
            {"role": "user", "content": "Olá"},
            {"role": "assistant", "content": "Olá! Como posso ajudá-lo?"}
        ]
    )

    # 7. Testes de validação (campo 'question' ausente)
    test_missing_question_field()

    print("\n" + "="*60)
    print("TESTES CONCLUÍDOS")
    print("="*60)
