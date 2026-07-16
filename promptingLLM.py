import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from openai import OpenAI


def openai_create_knowledge(text, culture, dimension):
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


    prompt = f"""
    You are an expert assistant in cultural text analysis.
    Your task is to read the following texts provided and extract only the relevant information related to the {culture} culture.
    Just return one single result for all text provided and return result in a structured format with the following fields:

    - title: a short title summarizing the feature.
    - culture": {culture},
    - dimension": {dimension},
    - snippet: the original text supporting this dimension. If there are multiple texts, you put one snippet for each text, separated by //. Keep the original text as is, including language. Do not modify it
    - Knowledge: a concise summary in english of the information related to the {dimension} dimension in the {culture} culture, based solely on the provided text.  

    Important decision rule:

    If the provided text contains direct or indirect evidence about the dimension "{dimension}" in the culture "{culture}", then you must write a knowledge summary based on that evidence.

    Do not require deep cultural interpretation, historical explanation, or broad social context. A concrete behavioral description is enough.

    Only write "Not enough information." in the knowledge field if the text does not mention the dimension, does not describe the behavior, or does not allow any reasonable conclusion about "{dimension}" in "{culture}"

    If you write "Not enough information.", the title must be:
    "{dimension} - {culture} (info missing)"

    In that case, also explain briefly in the knowledge field why the text was insufficient.

    Do not invent information.
    Do not use information outside the provided text.
    Just create one Knowledge text.
    
    Texts to analyze:
    {prompt_texts}

    """
    load_dotenv()

    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content


def interweb_create_knowledge(args, text, culture, dimension):
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
    """
    model = args.llm_model  # Replace with the model available in your API. gpt-4o-mini
    payload = {
        "model": model,  # Replace with the model available in your API. gpt-4o-mini
        "messages": [
            {
                "role": "system",
                "content": f"You are an expert assistant in cultural text analysis. Your task is to read the following texts provided and extract only the relevant information related to the {culture} culture."
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n\n{prompt_texts}"
            }
        ]
    }

    response = requests.post(
        f"{url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    print("Interweb API is using:", model)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
    else:
        return response.json()["choices"][0]["message"]["content"]
 

def interweb_model_list(args, text, question_type, culture, question_language, model= "gpt-4o-mini"):
    API_KEY = "yMbyBst2N4RBPIY8UJAxMFBdzUiaLM1bBoskkitspjxmszNcva8IkKb8tO0OHI0C"
    url = "https://interweb.l3s.uni-hannover.de"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    response = requests.get(
        f"{url}/models",
        headers=headers,
        timeout=60
    )

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

