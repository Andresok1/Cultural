import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://interweb.l3s.uni-hannover.de/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {os.getenv('INTERWEB_API_KEY')}",
    "Content-Type": "application/json",
    "accept": "application/json"
}

payload = {
    "model": "gpt-4.1-mini",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ]
}

print("Sending request...")

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60
)

print("\nStatus code:")
print(response.status_code)

print("\nResponse:")
print(response.text)