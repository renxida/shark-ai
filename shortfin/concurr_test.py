import concurrent.futures
import requests
import sys

url = "http://localhost:8003/generate"
payload = {
    "text": "1 2 3 4 5 ",
    "sampling_params": {"max_completion_tokens": 50},
}


def fetch(url, payload):
    return requests.post(url, json=payload)


if __name__ == "__main__":
    # Get number of workers from command line, default to 2 if not provided
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch, url, payload) for _ in range(num_workers)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result.status_code, result.text)
