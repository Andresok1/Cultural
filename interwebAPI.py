import requests
import json

def interweb_knowledge_prompting(text, culture, dimension):

    API_KEY = "yMbyBst2N4RBPIY8UJAxMFBdzUiaLM1bBoskkitspjxmszNcva8IkKb8tO0OHI0C"

    url = "https://interweb.l3s.uni-hannover.de"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    user_prompt = f"""
    Just return one single result for all text provided and return result in a structured format with the following fields:

    - title: a short title summarizing the feature.
    - culture": {culture},
    - dimension": {dimension},
    - snippet: the original text supporting this feature. If there are multiple texts, you put one snippet for each text, separated by //.
    - Knowledge: a concise summary of the information related to the {dimension} dimension in the {culture} culture, based solely on the provided text.

    If the text **does not contain enough information about the culture**, do not invent anything and say "Not enough information." in the knowledge field, but fill the title with the dimension and culture and added a (info missing), so it can be tracked. Also say why you decide to write "not enought information"

    """

    payload = {
        "model": "gpt-4o-mini",  # Replace with the model available in your API
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

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
    else:
        return response.json()["choices"][0]["message"]["content"]
    


    # print("user_prompt:", user_prompt)

    # search_response = requests.get(
    #     f"{url}/search",
    #     headers=headers,
    #     params=params,
    #     timeout=30
    # )


    # print("Final URL:", search_response.url)
    # print("Status code:", search_response.status_code)


    # if search_response.status_code != 200:
    #     print("Error:", search_response.text)
    # else:
    #     data = search_response.json()
    #     print("JSON keys:", data.keys())

    #     try:
    #         results = data["results"][0]["items"]
    #         top_3 = results[:3]

    #         for i, item in enumerate(top_3, start=1):
    #             print(f"\nDocument {i}:")
    #             print("Rank:", item.get("rank"))
    #             print("Title:", item.get("title"))
    #             print("URL:", item.get("url"))
    #             print("Description:", item.get("description"))
    #             print("Authors:", item.get("authors"))
    #             print("Date:", item.get("date"))
    #             print("Type:", item.get("type"))
    #             print("Service:", item.get("service"))

    #     except Exception as e:
    #         print("Error extracting documents:", e)


# interweb_knowledge_prompting(doc, "Colombian", "names examples ")
