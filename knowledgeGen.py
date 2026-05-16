from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API key initialization
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def create_knowledge(text, culture, dimension):
    """
    Extracts important features and content related to a specific culture
    from a given text. Returns format: title, original snippet and knowledge extrated from the text.
    Does not invent information if there is insufficient support.
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
    - snippet: the original text supporting this feature.
    - Knowledge: a concise summary of the information related to the {dimension} dimension in the {culture} culture, based solely on the provided text.

    If the text **does not contain enough information about the culture**, do not invent anything and say "Not enough information."

    Texts to analyze:
    {prompt_texts}

    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content