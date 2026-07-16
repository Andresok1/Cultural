import json
import os
import re

from dotenv import load_dotenv
from questionGen import csv_saver
from promptingLLM import interweb_create_knowledge, openai_create_knowledge
from deep_translator import GoogleTranslator

def translate(text, target_lang):
    return GoogleTranslator(source="en", target=target_lang).translate(text)


def OpenAI_create_knowledge(text, culture, dimension):
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
    load_dotenv()

    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content


def knowledge_level_manager(args, timestamp, query_results):
    """
    It manages between atomic and collective to organize knowledge generation.
    """

    resultados = [] 


    culture_dfs = {} 
    
    for key, info in query_results.items():
        culture = info.get('culture', [])
        dimension = info.get('dimension')
        languages  = info.get('languages', [])

        if culture not in culture_dfs:
            culture_dfs[culture] = []

        output_path = f"results/knowledge_output.json"
        input_path = f"results/knowledge_input.json"

        if os.path.exists(output_path): #Update for knowledge_output
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    knowledge_output_dict = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {output_path} is invalid JSON. Reinitializing.")
                knowledge_output_dict = {}
        else:
            knowledge_output_dict = {}     #Structure in Knowledge_output.json

        if os.path.exists(input_path):  #Update for knowledge_input
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    knowledge_input_dict = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {input_path} is invalid JSON. Reinitializing.")
                knowledge_input_dict = {}
        else:
            knowledge_input_dict = {}

        knowledge_output= []    #For each Query
        knowledge_input= []
        
        count = 0
        
        if args.knowledge_level == "atomic":
            for lang, data in languages.items():
                query_by_language = data.get("query", [])
                ranking = data.get("ranking", [])
                
                count_by_language = 0

                for content in ranking:
                    if content:
                        knowledge_input.append(content) 
                        if args.api == "openai":
                            knowledge_text= openai_create_knowledge(text=content, culture=culture, dimension=dimension)
                        else:
                            knowledge_text = interweb_create_knowledge(text=content, culture=culture, dimension=dimension)#
                            #IF here it says something about (info missing) it should look for more docs

                        knowledge_output.append(knowledge_text)   
                        count_by_language += 1

                    if count_by_language == 3:
                        break


                print(f"For key {key} in {lang}:")
                print(f"you got {count_by_language} documents")

                count += count_by_language
            
            print(f"After all languages the total Counter for {key} is now: {count}")

            if count < args.max_results:
                print(f"DOCS MISSING {count}/{args.max_results}")

        else: #"collective" knowledge level
            for lang, data in languages.items():
                query_by_language = data.get("query", [])
                ranking = data.get("ranking", [])

                count_by_language = 0
                knowledge_input_cache = [] # Declaration and reset knowledge_input for the next language
                for content in ranking:
                    if content:
                        knowledge_input.append(content) #Accumulation of Knowlege input for collective analysis. 
                        count_by_language += 1

                    if count_by_language == 3:
                        break
                
                print(f"For key {key} in {lang}:")
                print(f"you got {count_by_language} documents")

                count += count_by_language

            if count < args.max_results:
               print(f"DOCS MISSING {count}/{args.max_results}")

            if args.api == "openai":
                knowledge_text= OpenAI_create_knowledge(text=knowledge_input, culture=culture, dimension=dimension)
            else:
                knowledge_text = interweb_knowledge_prompting(text=knowledge_input, culture=culture, dimension=dimension)

            if knowledge_text is None:
                knowledge_text = ""

            knowledge_text_cleaned = re.sub(
                r"^```json\s*|\s*```$",  # quitar ```json al inicio y ``` al final
                "",
                knowledge_text,
                flags=re.DOTALL
            ).replace(";", ",").strip()

            knowledge_output.append(knowledge_text_cleaned)    ###One knowledge result for all docs

        if culture not in knowledge_output_dict:
            knowledge_output_dict[culture] = {}

        if culture not in knowledge_input_dict:
            knowledge_input_dict[culture] = {}

        knowledge_output_dict[culture][dimension] = knowledge_output   

        knowledge_input_dict[culture][dimension] = knowledge_input 

        with open(output_path, "w", encoding="utf-8") as f:     #Save knowledge_output
            json.dump(knowledge_output_dict, f, ensure_ascii=False, indent=2)

        with open(input_path, "w", encoding="utf-8") as f:      #Save knowledge_input
            json.dump(knowledge_input_dict, f, ensure_ascii=False, indent=2)

        csv_saver(args, dimension, culture, timestamp, culture_dfs, output_path)


    return knowledge_output

