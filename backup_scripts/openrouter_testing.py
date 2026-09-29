import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    "qwen/qwen3.5-9b",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "cohere/north-mini-code:free",
    "liquid/lfm-2.5-2.6b:free",
]

TEST_PROMPT = """
Extract cultural knowledge from the following text.

Text:
"In Colombia, guests are usually offered coffee when visiting someone's home.
The host often prepares the drink as a sign of hospitality."

Return a concise structured explanation.
"""


client = OpenAI(
    base_url=BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=180
)


for model in MODELS:

    print("\n" + "=" * 60)
    print(f"Testing model: {model}")
    print("=" * 60)

    start = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a cultural knowledge extraction assistant."
                },
                {
                    "role": "user",
                    "content": TEST_PROMPT
                }
            ]
        )

        elapsed = time.perf_counter() - start

        answer = response.choices[0].message.content

        print("SUCCESS")
        print(f"Time: {elapsed:.2f}s")
        print(answer[:300])


    except Exception as e:

        elapsed = time.perf_counter() - start

        print("FAILED")
        print(f"Time: {elapsed:.2f}s")
        print(e)