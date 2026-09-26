import os
import re
from urllib import response
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from time import perf_counter
import time
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


ROLE_KNOWLEDGE = f"You are an expert assistant in cultural text analysis. Your task is to read the following texts provided and extract only the relevant information related to the culture."

def PROMPT_KNOWLEDGE(culture, dimension, prompt_texts):
    user_prompt = f"""
        You will be provided with a collection of documents related to a specific culture and dimension.

        Your task is to analyze all documents together as a single knowledge source and identify the most relevant pieces of information related to the given culture and dimension.

        Extract all distinct knowledge entries supported by relevant evidence in the provided documents. Do not require a minimum number of entries. Only create an entry when the documents contain meaningful information directly related to the requested dimension and culture.
        Relevant evidence describes a practice, expectation, behavior, or fact concerning the requested dimension in the requested cultural context. Mentioning the culture alone, or discussing an adjacent topic without an explicit connection to the dimension, is not sufficient.
        A finding may address only one specific aspect of the dimension; it does not need to describe the entire culture. Preserve any limits stated in the source, such as location, setting, group, or uncertainty. Do not turn a specific example into a general cultural rule.

        For each relevant finding, return:

        - title: A short title summarizing the main idea or cultural feature.
        - culture: {culture}
        - dimension: {dimension}
        - snippet: The exact original text extracted from the provided documents that supports this finding. Do not modify, translate, summarize, or paraphrase the snippet. Preserve the original language and wording.
        - knowledge: A concise summary in English explaining what can be learned from this snippet about the {dimension} dimension in {culture} culture. The knowledge summary must be based exclusively on the provided snippet.

        Important rules:
        - Treat all provided documents as one combined source. Do not analyze documents independently.
        - Search across the entire document collection to identify distinct findings. When multiple documents describe the same cultural feature, avoid duplicate entries and select a supporting excerpt from one document for that finding.
        - Each snippet must be one continuous excerpt with enough context to support the knowledge summary. Do not stitch passages together or infer explanations that the excerpt does not provide.
        - Each snippet and knowledge pair must represent one distinct and meaningful cultural finding.
        - If multiple unrelated important findings exist, create separate entries for each one.
        - Do not create redundant entries describing the same information.
        - Do not add external knowledge or assumptions beyond the provided documents.
        - If no relevant information is found for the requested culture and dimension, write as title: (NOT RELEVANT INFORMATION) and the rest as "EMPTY" .

        Return the output only in the following structured JSON format:

        [
            {{
                "title": "...",
                "culture": "{culture}",
                "dimension": "{dimension}",
                "snippet": "...",
                "knowledge": "..."
            }}
        ]
        Documents:
        {prompt_texts}

        Do not introduce your answer with any text or explanation. Only return the JSON array of knowledge entries.
        """
    return user_prompt


def openai_create_knowledge(args, text, culture, dimension, language=None):
    """
    Extracts important features and content related to a specific culture using OpenAI API
    from a given text. Returns format: title, original snippet and knowledge extrated from
    the text. Does not invent information if there is insufficient support.
    """
    openai_model = "gpt-4o-mini"  # Use the model specified in the command-line arguments

    if not text:
        print("No documents were given to produce a knowledge entry.")
        return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    load_dotenv()

    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )

    print(f"{language + ' | ' if language else ''}openai / {openai_model} | Sending request...", flush=True)
    started = perf_counter()

    response = client.chat.completions.create(
        model= openai_model, #OPENAI constant Model
        messages=[
            {
                "role": "user", 
                "content": ROLE_KNOWLEDGE + user_prompt
            }
        ]
    )

    content = response.choices[0].message.content
    print(f"{language + ' | ' if language else ''}openai / {openai_model} | Finished after {perf_counter() - started:.1f}s")
    return content


def interweb_create_knowledge(args, text, culture, dimension, retries = 1, language=None):
    """
    Extracts important features and content related to a specific culture using Interweb API
    from a given text. Returns format: title, original snippet and knowledge extrated from
    the text. Does not invent information if there is insufficient support.
    """
    interweb_model= "gpt-4.1-mini"
    load_dotenv()
    INTERWEB_API_KEY = os.getenv("INTERWEB_API_KEY")


    headers = {
        "Authorization": f"Bearer {INTERWEB_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }

    if not text:
        print("No documents were given to produce a knowledge entry.")
        return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    payload = {
        "model": interweb_model,  # Replace with the model available
        "messages": [
            {
                "role": "system",
                "content": ROLE_KNOWLEDGE,
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n"
            }
        ]
    }

    try:
        print(f"{language + ' | ' if language else ''}interweb / {interweb_model} | Sending request...", flush=True)
        started = perf_counter()
        response = requests.post(
            f"https://interweb.l3s.uni-hannover.de/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            print("Error:", response.status_code, response.text)

            if retries > 0:
                print(f"Retrying... attempts left: {retries}")
                return interweb_create_knowledge(
                    args,
                    text,
                    culture,
                    dimension,
                    retries - 1, 
                    language=language
                )
            else: 
                print("Game Over")
            
            return None
        
        answer = response.json()["choices"][0]["message"]["content"]

        if answer is None or "[]" in answer or answer.strip() == "":
            if retries > 0:
                print(f"Empty response. Retrying... attempts left: {retries}")
                return interweb_create_knowledge(
                    args,
                    text,
                    culture,
                    dimension,
                    retries - 1, 
                    language=language
                )

            return None

        print(f"{language + ' | ' if language else ''}interweb / {interweb_model} | Finished after {perf_counter() - started:.1f}s")
        return answer
    
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

        if retries > 0:
            return interweb_create_knowledge(
                args,
                text,
                culture,
                dimension,
                retries - 1, 
                language=language
            )

        return None

def openrouter_create_knowledge(args, text, culture, dimension, retries=5, language=None):

    openrouter_model = "openai/gpt-4o"
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    
    if not text:
        print(f"No documents were given to produce a knowledge entry-> {culture}, {dimension}.")
        return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    data = {
        "model": f"{openrouter_model}",
        "messages": [
            {
                "role": "system",
                "content": ROLE_KNOWLEDGE,
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n"
            }
        ]
    }

    print(f"{language + ' | ' if language else ''}openrouter / {openrouter_model} | Sending request...", flush=True)
    started = perf_counter()
    response = requests.post(
        f"{url}/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()

    response_json = response.json()

    # print("OPENROUTER RESPONSE:")
    # print(response_json)

    answer = response.json()["choices"][0]["message"]["content"]
    print(f"{language + ' | ' if language else ''}openrouter / {openrouter_model} | Finished after {perf_counter() - started:.1f}s")

    if not answer or not answer.strip():
        print("WARNING: Empty answer from OpenRouter")
        return None


    return answer


def inference_create_knowledge(args, text=None, culture=None, dimension=None, retries=2, language=None):
    inference_model = "granite-4.1:8b-bf16"

    load_dotenv()
    INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY")

    if not text:
        print(f"No documents were given to produce a knowledge entry-> {culture}, {dimension}.")
        return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    client = OpenAI(
        base_url=f"https://inference.kbs.uni-hannover.de/v1",
        api_key=os.getenv("INFERENCE_API_KEY"),
        timeout=300,
    )

    for attempt in range(retries + 1):

        try:

            print(
                f"{language + ' | ' if language else ''}"
                f"inference / {inference_model} | "
                f"Sending request "
                f"(attempt {attempt + 1}/{retries + 1})...",
                flush=True
            )

            started = perf_counter()

            response = client.chat.completions.create(
                model=inference_model,
                messages=[
                    {
                        "role": "system",
                        "content": ROLE_KNOWLEDGE
                    },
                    {
                        "role": "user",
                        "content": f"{user_prompt}\n"
                    }
                ]
            )

            answer = response.choices[0].message.content

            print(
                f"{language + ' | ' if language else ''}"
                f"inference / {inference_model} | "
                f"Finished after "
                f"{perf_counter() - started:.1f}s"
            )

            if (
                answer is None
                or "[]" in answer
                or answer.strip() == ""
            ):

                print("Empty response from inference.")

                if attempt < retries:
                    wait_time = min(2 ** attempt, 60)

                    print(f"Retrying in {wait_time}s...")
                        

                    time.sleep(wait_time)
                    continue

                print(
                    f"Game Over: empty response for "
                    f"{culture} | {dimension} | {language}"
                )

                return None

            return answer

        except (
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError
        ) as e:

            print(
                f"API error on attempt "
                f"{attempt + 1}/{retries + 1}:"
            )

            print(
                f"Timeout after {perf_counter()-started:.1f}s"
            )

            print(f"{type(e).__name__}: {e}")

            if attempt >= retries:

                print(f"Game Over: {culture} | {dimension} | {language}")

                return None

            wait_time = min(15 * (2 ** attempt), 60)

            print(f"Retrying in {wait_time}s...")                

            time.sleep(wait_time)

        except Exception as e:

            print(
                f"Unexpected error: "
                f"{type(e).__name__}: {e}"
            )

            return None

    return None

def interweb_model_list():
    API_KEY = "yMbyBst2N4RBPIY8UJAxMFBdzUiaLM1bBoskkitspjxmszNcva8IkKb8tO0OHI0C"
    url = "https://interweb.l3s.uni-hannover.de"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    model_name = "gpt-5-nano" #TEST

    response = requests.get(
        f"{url}/v1/models",
        headers=headers,
        timeout=60
    )
    print(response.json())
    response.raise_for_status()
    models = response.json()

    df = models_to_table(models)
    df.to_csv("models.csv", index=False, encoding="utf-8")
    print(df)

def interweb_llm_tester(args):

    """ 
    This function is a tester for the interweb LLM API. It sends a prompt to the API and prints the response.
    """
    role = "you are a Colombian in the Moon"

    prompt= "Tell me a short history about you"

    load_dotenv()
    INTERWEB_API_KEY = os.getenv("INTERWEB_API_KEY")
    url = "https://interweb.l3s.uni-hannover.de"
    
    headers = {
        "Authorization": f"Bearer {INTERWEB_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }


    payload = {
        "model": args.llm_model, #Model can be changed.
        "messages": [
            {
                "role": "system",
                "content": role,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }

    response = requests.post(
        f"{url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
    else:
        return response.json()["choices"][0]["message"]["content"]


def models_to_table(response):
    df = pd.DataFrame(response["data"])

    if "price" in df.columns:
        price_df = pd.json_normalize(df["price"])
        price_df.columns = ["price_" + c for c in price_df.columns]
        df = df.drop(columns=["price"]).join(price_df)

    return df

def json_cleanig(text):
    """
    Extract JSON array or object from LLM response.
    """

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

    array_match = re.search(r"\[.*\]", text, re.DOTALL)

    if array_match:
        return array_match.group(0)

    object_match = re.search(r"\{.*\}", text, re.DOTALL)

    if object_match:
        return object_match.group(0)

    return None


def openrouter_testing(llm_model):

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    
    data = {
        "model": f"{llm_model}",  # Replace with the model available in your API. gpt-4o-mini
        "messages": [
            {
                "role": "system",
                "content": "you are colombian in the Moon",
            },
            {
                "role": "user",
                "content": "HALLooo "
            }
        ]
    }


    started = perf_counter()
    response = requests.post(
        f"{url}/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]


    if not answer or not answer.strip():
        print("WARNING: Empty answer from OpenRouter")
        return None

    return answer