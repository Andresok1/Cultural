from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API key initialization
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
def extract_features(text, culture, dimension):
    """
    Extracts important features and content related to a specific culture
    from a given text. Returns format: title, original snippet, paragraph if possible.
    Does not invent information if there is insufficient support.
    """
    
    prompt = f"""
    You are an expert assistant in cultural text analysis.
    Your task is to read the following text and extract only the relevant information related to the culture: "{culture}".
    For each relevant point, return:

    - title: a short title summarizing the feature.
    - snippet: the original text supporting this feature.
    - paragraph: if possible, indicate the paragraph number (1 being the first).

    If the text **does not contain enough information about the culture**, do not invent anything and say "Not enough information."

    Text to analyze:
    {text}
    Response format:
    [
    {{

        "title": "...",
        "dimension": {dimension},
        "snippet": "...",
        "paragraph": ...
    }},
    ...
    ]
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content

# Example usage



if __name__ == "__main__":


    ruta_archivo = r"C:\Users\Andres\Repos\Cultural_thesis\results.json"

    resultados = [] 

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        data = json.load(f)

    for dimension, items in data.items():
        for item in items:

            title = item.get("title", "")
            link = item.get("link", "")
            content = item.get("content", "")

            resultados.append({
                "dimension": dimension,
                "title": title,
                "link": link,
                "content": content
            })

    test = resultados[0] if resultados else None

    test_dimension = test["dimension"] if test else "Fetch error"
    test_culture = "Colombian"  # Example culture, you can modify this as needed
    test_content = test["content"] if test else "Fetch error"



    with open("knowledge_input.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    result = extract_features(text=test_content, culture=test_culture, dimension=test_dimension)

    print(result)