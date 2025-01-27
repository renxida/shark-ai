import requests
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_text(prompt: str, port: int = 8000) -> str:
    """Make a generation request to the server.

    Args:
        prompt: Input text prompt
        port: Server port number (default: 8000)

    Returns:
        Generated text response
    """
    response = requests.post(
        f"http://localhost:{port}/generate",
        headers={"Content-Type": "application/json"},
        json={
            "text": prompt,
            "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
            "rid": uuid.uuid4().hex,
            "stream": False,
        },
    )
    response.raise_for_status()

    # Parse response format
    data = response.text
    if data.startswith("data: "):
        return data[6:].rstrip("\n")
    return data


def main():
    # Parameters
    prompt = "1 2 3 4 5 "  # Same prompt used in tests
    concurrent_requests = 9

    # Make concurrent requests using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [
            executor.submit(generate_text, prompt) for _ in range(concurrent_requests)
        ]

        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            try:
                response = future.result()
                print(f"Response {i}: {response}")
            except Exception as e:
                print(f"Request {i} failed: {str(e)}")


if __name__ == "__main__":
    main()
