import os
import re
from urllib import response
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from openai import OpenAI


ROLE_KNOWLEDGE = f"You are an expert assistant in cultural text analysis. Your task is to read the following texts provided and extract only the relevant information related to the culture."

def PROMPT_KNOWLEDGE(culture, dimension, prompt_texts):
    user_prompt = f"""
        You will be provided with a collection of documents related to a specific culture and dimension.

        Your task is to analyze all documents together as a single knowledge source and identify the most relevant pieces of information related to the given culture and dimension.

        Generate between 1 and 5 independent knowledge entries. Only create an entry when the documents contain meaningful information directly related to the requested dimension and culture.

        For each relevant finding, return:

        - title: A short title summarizing the main idea or cultural feature.
        - culture: {culture}
        - dimension: {dimension}
        - snippet: The exact original text extracted from the provided documents that supports this finding. Do not modify, translate, summarize, or paraphrase the snippet. Preserve the original language and wording.
        - knowledge: A concise summary in English explaining what can be learned from this snippet about the {dimension} dimension in {culture} culture. The knowledge summary must be based exclusively on the provided snippet.

        Important rules:
        - Treat all provided documents as one combined source. Do not analyze documents independently.
        - Search across the entire document collection and combine information when multiple documents describe the same cultural feature.
        - Each snippet and knowledge pair must represent one distinct and meaningful cultural finding.
        - If multiple unrelated important findings exist, create separate entries for each one.
        - Do not create redundant entries describing the same information.
        - Do not add external knowledge or assumptions beyond the provided documents.
        - If no relevant information is found for the requested culture and dimension, return an empty list.

        Return the output only in the following structured JSON format:

        [
            {{
                "title": "...",
                "culture": "{culture}",
                "dimension": "{dimension}",
                "snippet": "...",
                "knowledge": "..."
                "snippet": "...",
                "knowledge": "..."
                ...
            }}
        ]
        Documents:
        {prompt_texts}

        Do not introduce your answer with any text or explanation. Only return the JSON array of knowledge entries.
        """
    return user_prompt


def openai_create_knowledge(args, text, culture, dimension):
    """
    Extracts important features and content related to a specific culture using OpenAI API
    from a given text. Returns format: title, original snippet and knowledge extrated from
    the text. Does not invent information if there is insufficient support.
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    load_dotenv()

    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model= "gpt-4o-mini", #OPENAI constant Model
        messages=[{"role": "user", "content": ROLE_KNOWLEDGE + user_prompt}]
    )

    content = response.choices[0].message.content
    return content


def interweb_create_knowledge(args, text, culture, dimension, retries = 5):
    """
    Extracts important features and content related to a specific culture using Interweb API
    from a given text. Returns format: title, original snippet and knowledge extrated from
    the text. Does not invent information if there is insufficient support.
    """
    
    load_dotenv()
    INTERWEB_API_KEY = os.getenv("INTERWEB_API_KEY")

    url = "https://interweb.l3s.uni-hannover.de"

    headers = {
        "Authorization": f"Bearer {INTERWEB_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    print("     Interweb API is using:", args.llm_model)

    payload = {
        "model": args.llm_model,  # Replace with the model available in your API. gpt-4o-mini
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
            
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
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
                    retries - 1
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
                    retries - 1
                )

            return None

        print("     DONE: knowledge created")
        return answer
    
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

        if retries > 0:
            return interweb_create_knowledge(
                args,
                text,
                culture,
                dimension,
                retries - 1
            )

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
    