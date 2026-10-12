import os
import time
from time import perf_counter

import requests
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

ROLE_STUDENT = "you are a colombian in the moon"

def call_llm(
    endpoint,
    llm_model,
    prompt,
    retries=2,
    status_label="answering"
):
    """
    Generic LLM caller for OpenAI, OpenRouter, Interweb and Inference endpoints.
    Only the prompt/model changes when calling this function.
    """

    load_dotenv()

    configs = {

        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            }
        },

        "openrouter": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
            }
        },

        "interweb": {
            "url": "https://interweb.l3s.uni-hannover.de/v1/chat/completions",
            "api_key": os.getenv("INTERWEB_API_KEY"),
            "headers": {
                "Authorization": f"Bearer {os.getenv('INTERWEB_API_KEY')}",
                "accept": "application/json",
                "Content-Type": "application/json"
            }
        },

        "inference": {
            "url": "https://inference.kbs.uni-hannover.de/v1",
            "api_key": os.getenv("INFERENCE_API_KEY")
        }
    }


    config = configs[endpoint]


    for attempt in range(retries + 1):

        try:

            print(
                f"{'ANSWERING':<9} | "
                f"{status_label:<10} | "
                f"{endpoint} / {llm_model} | "
                f"Attempt {attempt + 1}/{retries + 1}",
                flush=True
            )


            started = perf_counter()


            # OpenAI compatible HTTP endpoints
            if endpoint in ["openai", "openrouter", "interweb"]:


                payload = {
                    "model": llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": ROLE_STUDENT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }


                response = requests.post(
                    config["url"],
                    headers=config["headers"],
                    json=payload,
                    timeout=60
                )


                if response.status_code == 429:

                    wait_time = min(2 ** attempt, 60)

                    print(
                        f"Rate limit. Waiting {wait_time}s..."
                    )

                    time.sleep(wait_time)
                    continue


                if response.status_code != 200:

                    print(
                        response.status_code,
                        response.text
                    )

                    continue


                answer = response.json()["choices"][0]["message"]["content"]



            # Inference endpoint using OpenAI client
            elif endpoint == "inference":


                client = OpenAI(
                    base_url="https://inference.kbs.uni-hannover.de/v1",
                    api_key=config["api_key"],
                    timeout=60
                )


                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": ROLE_STUDENT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


                answer = response.choices[0].message.content



            else:

                raise ValueError(
                    f"Unknown endpoint: {endpoint}"
                )



            if (
                answer is None
                or answer.strip() == ""
                or "[]" in answer
            ):

                raise ValueError(
                    "Empty response"
                )


            print(
                f"{'ANSWERING':<9} | "
                f"{status_label:<10} | "
                f"{endpoint} / {llm_model} | "
                f"Finished after {perf_counter()-started:.1f}s"
            )


            return answer



        except (
            requests.exceptions.RequestException,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            ValueError
        ) as e:


            print(
                f"Error attempt {attempt+1}/{retries+1}: "
                f"{type(e).__name__}: {e}"
            )


            if attempt < retries:

                wait_time = min(15 * (2 ** attempt), 60)

                print(
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

            else:

                print(
                    f"Game Over: {endpoint} / {llm_model}"
                )

                return None


GRADER_MODELS = [
    # ("openai", "gpt-4o"),
    # ("interweb", "gpt-4.1-mini"),
    # ("openrouter", "qwen/qwen3.5-9b"),


    # ("inference", "granite-4.1:8b-bf16"),
    # ("inference", "apertus:8b-2509-it-bf16"),
    # ("inference", "qwen3-vl:8b"),
    

    # ("inference", "gpt-oss:120b-mxfp4"),
]

# paila
    # ("inference", "qwen3:8b"),
    # ("inference", "gpt-oss:20b-mxfp4"),
    # ("inference", "gemma4:e4b-q8_0"),
    # ("inference", "ideogram-4:5b-nf4"),
    # ("inference", "gemma4:e4b-q8_0"),


for endpoint, model in GRADER_MODELS:
    print(endpoint, model)
    answer= call_llm(endpoint, model, "tell your secret to be like you",retries=2)
    print(answer)

