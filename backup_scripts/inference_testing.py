import os
import time
import json
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://inference.kbs.uni-hannover.de/v1"

MODELS = [
    "granite-4.1:8b-bf16",
    "gemma3:4b-f16",
    "qwen3.5:9b-bf16",
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
    api_key=os.getenv("INFERENCE_API_KEY"),
    timeout=180
)


results = []


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

        result = {
            "model": model,
            "status": "success",
            "time_seconds": round(elapsed, 2),
            "output_length": len(answer),
            "response": answer
        }

        print("SUCCESS")
        print(f"Time: {elapsed:.2f}s")
        print(f"Output length: {len(answer)} characters")
        print(answer[:300])


    except Exception as e:

        elapsed = time.perf_counter() - start

        result = {
            "model": model,
            "status": "failed",
            "time_seconds": round(elapsed, 2),
            "error": str(e)
        }

        print("FAILED")
        print(f"Time: {elapsed:.2f}s")
        print(e)


    results.append(result)


filename = f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)


print("\nFinished.")
print(f"Results saved to {filename}")