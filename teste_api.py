import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


def main() -> None:
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url, timeout=5)
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


if __name__ == "__main__":
    main()
