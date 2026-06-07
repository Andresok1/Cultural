from openai import OpenAI
import json
import os
from dotenv import load_dotenv

from questionGen import csv_saver


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API key initialization
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def create_knowledge(text, culture, dimension):
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
    - snippet: the original text supporting this feature. If there are multiple texts, you put one snippet for each text, separated by //.
    - Knowledge: a concise summary of the information related to the {dimension} dimension in the {culture} culture, based solely on the provided text.

    If the text **does not contain enough information about the culture**, do not invent anything and say "Not enough information." in the knowledge field, but fill the title with the dimension and culture and added a (info missing), so it can be tracked.

    Texts to analyze:
    {prompt_texts}

    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content


def knowledge_level_manager(args, query_results):
    """
    It manages between atomic and collective to organize knowledge generation.
    """

    resultados = [] 

    print(f"--- Analazing query documents ---")
    with open(query_results, "r", encoding="utf-8") as f:
        data = json.load(f)

    culture_dfs = {} 
    
    for query, info in data.items():
        dimension = info.get('dimension')
        culture = info.get('culture', [])
        docs = info.get('result', [])

        if culture not in culture_dfs:
            culture_dfs[culture] = []

        
        knowledge_output= []

        count = 0

        if args.knowledge_level == "atomic":
            for doc in docs:
                content = doc.get('content')   #sacar el create_knowledge del if
                if content and len(content.strip()) > 300:     
                    kl= create_knowledge(text=content, culture=culture, dimension=dimension)
                    knowledge_output.append(kl)   
                    count += 1
                
                if count >= args.max_results:
                    break
            if count < args.max_results:
                print(f"Warning: Only {count} results with content found for query '{query}' (less than the max of {args.max_results})")

        else: #"collective" knowledge level
            knowledge_input= []
            for doc in docs:
                content = doc.get('content')

                if content:
                    knowledge_input.append({content})
                    count += 1

                if count >= args.max_results:
                    break
                
            if count < args.max_results:
                print(f"Warning: Only {count} results with content found for query '{query}' (less than the max of {args.max_results})")
            
            kl= create_knowledge(text=knowledge_input, culture=culture, dimension=dimension)
            kl_cleaned = kl.replace(";", ",").strip() 
            knowledge_output.append(kl)    ###One knowledge result for all docs


        with open(f"results/knowledge_output_{culture}.json", "w", encoding="utf-8") as f:
            json.dump(knowledge_output, f, ensure_ascii=False, indent=2)


        knowledge_path = rf"C:\Users\Andres\Repos\Cultural_thesis\results\knowledge_output_{culture}.json"

        csv_saver(args, dimension, culture, culture_dfs, knowledge_path)


    return knowledge_output

