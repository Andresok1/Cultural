import json
import os

from questionGen import csv_saver
from promptingLLM import interweb_create_knowledge, json_cleanig, openai_create_knowledge, openrouter_create_knowledge
from deep_translator import MyMemoryTranslator

def translate(text, target_lang):
    try:
        answer= MyMemoryTranslator(
            source="english",
            target=target_lang
        ).translate(text) 
        # print(answer)
        return answer
    except Exception as e:
        print(e)
        return text


def knowledge_level_manager(args, timestamp, query_results):
    """
    It manages between atomic and collective to organize knowledge generation.
    """

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
        knowledge_input_dicc = {}
        knowledge_output_dicc = {}
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
                            knowledge_text= openai_create_knowledge(args, text=content, culture=culture, dimension=dimension)
                        elif args.api == "openrouter":
                            knowledge_text = openrouter_create_knowledge(args, text=content, culture=culture, dimension=dimension)
                        else:
                            knowledge_text = interweb_create_knowledge(args, text=content, culture=culture, dimension=dimension) #Atomic
                            #IF here it says something about (info missing) it should look for more docs

                        knowledge_output.append(knowledge_text)   
                        count_by_language += 1

                    if count_by_language == 3:
                        break

                print("\n")
                print(f"For key {key} in {lang}:")

                count += count_by_language
            
            print(f"After all languages the total Counter for {key} is now: {count}")

            if count < args.max_results:
                print(f"DOCS MISSING {count}/{args.max_results}")

        else: #"collective" knowledge level
            print(f"###{key}'[# docs]:###")
            for lang, data in languages.items():
                print(f"- {lang}: {len(data.get('ranking', []))}")
            total = sum(len(data.get("ranking", [])) for data in languages.values())
            print(f"--------TOTAL: {total} --------\n")

            for lang, data in languages.items():
                query_by_language = data.get("query", [])
                ranking = data.get("ranking", [])

                valid_contents = [content for content in ranking if content]
                knowledge_input_cache = valid_contents[:max(0, args.max_results)]
                count_by_language = len(knowledge_input_cache)

                print(f"Knowledge | {culture} | {dimension} | {lang}")
#Collective
                if args.api == "openai":
                    knowledge_text= openai_create_knowledge(args, text=knowledge_input_cache, culture=culture, dimension=dimension)
                elif args.api == "openrouter":
                    knowledge_text = openrouter_create_knowledge(args, text=knowledge_input_cache, culture=culture, dimension=dimension)
                else:
                    knowledge_text = interweb_create_knowledge(args, text=knowledge_input_cache, culture=culture, dimension=dimension)

                if knowledge_text is None:
                    knowledge_text = ""

                knowledge_text_cleaned= json_cleanig(knowledge_text)

                model = "gpt-4o-mini" if args.api == "openai" else args.llm_model
                try:
                    entries = json.loads(knowledge_text_cleaned) if knowledge_text_cleaned else []
                    if isinstance(entries, dict):
                        entries = [entries]

                    valid_entries = [
                        entry for entry in entries
                        if isinstance(entry, dict)
                        and "NOT RELEVANT INFORMATION" not in str(entry.get("title", "")).upper()
                        and all(
                            isinstance(entry.get(field), str)
                            and entry[field].strip().upper() not in ("", "EMPTY")
                            for field in ("title", "snippet", "knowledge")
                        )
                    ]
                    knowledge_text_cleaned = json.dumps(valid_entries, ensure_ascii=False)
                    entry_count = len(valid_entries)
                    print(f"{args.api} / {model} | Knowledge entries: {entry_count}")

                except (json.JSONDecodeError, TypeError):
                    knowledge_text_cleaned = "[]"
                    print(f"{args.api} / {model} | Knowledge entries: unknown (invalid response)")

                knowledge_output.append(knowledge_text_cleaned)   ###One knowledge result by language
                knowledge_output_dicc[lang] = knowledge_text_cleaned

                count += count_by_language
                knowledge_input.append(knowledge_input_cache)   #Input storage for each language
                knowledge_input_dicc[lang] = knowledge_input_cache
                print()

            # print(f"({count}/{args.max_results}) documents as input in both languages") #(6/5) documents as input in both languages

            # knowledge_output.append(knowledge_text_cleaned)    ###One knowledge result for all docs

        if culture not in knowledge_output_dict:
            knowledge_output_dict[culture] = {}

        if culture not in knowledge_input_dict:
            knowledge_input_dict[culture] = {}

 
        knowledge_output_dict[culture][dimension] = knowledge_output_dicc

        knowledge_input_dict[culture][dimension] = knowledge_input_dicc


        with open(output_path, "w", encoding="utf-8") as f:     #Save knowledge_output (RAW knowledge)
            json.dump(knowledge_output_dict, f, ensure_ascii=False, indent=2)

        with open(input_path, "w", encoding="utf-8") as f:      #Save knowledge_input
            json.dump(knowledge_input_dict, f, ensure_ascii=False, indent=2)

        csv_saver(args, dimension, culture, timestamp, culture_dfs, knowledge_output_dict)


    return knowledge_output

