import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from typing import Optional, Dict, Any


def health_check(timeout=30) -> None:
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url, timeout=timeout)
        # Raise for non-2xx/3xx responses
        response.raise_for_status()
    except ConnectionError:
        print(f"Connection error when calling {url}. Is the server running?")
        return
    except Timeout:
        print(f"Request to {url} timed out.")
        return
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"HTTP error {status} from {url}: {e}")
        return
    except RequestException as e:
        print(f"Unexpected error calling {url}: {e}")
        return
    else:
        # Print status code and parsed content when possible
        print(response.status_code)
        content_type = response.headers.get("Content-Type", "")
        try:
            if "application/json" in content_type:
                print(response.json())
            else:
                print(response.text)
        except ValueError:
            # Fallback if JSON parsing fails unexpectedly
            print(response.text)


def send_question(
        question: str, 
        context: Optional[str] = None, 
        base_url: str = "http://localhost:8000", 
        timeout=1200
        ) -> Dict[str, Any]:
    """Send a question to the API (main.py) and return the JSON response.

    If context is provided, calls the full router endpoint '/route-query'.
    Otherwise, calls the simplified endpoint '/route-query/simple'.
    """
    endpoint = "/route-query" if context else "/route-query/simple"
    url = f"{base_url.rstrip('/')}{endpoint}"

    payload: Dict[str, Any] = {"question": question}
    if context:
        payload["context"] = context

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except ConnectionError:
        return {"error": f"Connection error calling {url}. Is the server running?"}
    except Timeout:
        return {"error": f"Request to {url} timed out."}
    except HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if getattr(e, "response", None) else ""
        return {"error": f"HTTP error {status} from {url}", "detail": detail}
    except RequestException as e:
        return {"error": f"Unexpected error calling {url}: {e}"}


if __name__ == "__main__":
    health_check()

    # Teste rota 






    # question = input("Faça sua pergunta:\n")
    # resposta = send_question(question).get("answer", "__Texto sem resposta__")
    # print(resposta)
