from result_paths import KNOWLEDGE_DIR
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

        output_path = KNOWLEDGE_DIR / "knowledge_output.json"
        input_path = KNOWLEDGE_DIR / "knowledge_input.json"

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
        
        print(f"###{key}'[# docs]:###")
        for lang, data in languages.items():
            print(f"- {lang}: {len(data.get('ranking', []))}")
        total = sum(len(data.get("ranking", [])) for data in languages.values())
        print(f"--------TOTAL: {total} --------\n")

        if args.knowledge_level == "atomic":
            for lang, data in languages.items():
                query_by_language = data.get("query", [])
                ranking = data.get("ranking", [])
                
                count_by_language = 0

                for document in ranking:
                    content = document.get("text", "") if isinstance(document, dict) else document
                    if content:
                        knowledge_input.append(document) 
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

        else: # collective: evaluate each batch within the existing language loop
            print(f"Knowledge | {culture} | {dimension}")
            if args.batch_size <= 0:
                raise ValueError("batch_size must be positive")

            for lang, data in languages.items():
                ranking = data.get("ranking", [])
                valid_contents = []
                for document in ranking:
                    if isinstance(document, dict):
                        content = document.get("text")
                    else:
                        content = document

                    if content:
                        valid_contents.append(document)
                language_entries = []
                total_batches = (len(valid_contents) + args.batch_size - 1) // args.batch_size  #upper bound division for total number of batches

                for offset in range(0, len(valid_contents), args.batch_size):
                    knowledge_input_cache = valid_contents[offset:offset + args.batch_size] #batch slider
                    prompt_texts = [document["text"] if isinstance(document, dict) else document
                                    for document in knowledge_input_cache]
                    batch_label = f"{lang} | Batch {offset // args.batch_size + 1}/{total_batches}"

                    if args.api == "openai":
                        knowledge_text = openai_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label)
                    elif args.api == "openrouter":
                        knowledge_text = openrouter_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label)
                    else:
                        knowledge_text = interweb_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label)

                    knowledge_text_cleaned = json_cleanig(knowledge_text or "")
                    try:
                        entries = json.loads(knowledge_text_cleaned) if knowledge_text_cleaned else []

                        if isinstance(entries, dict):
                            entries = [entries]

                        valid_entries = []

                        required_fields = ("title", "snippet", "knowledge")

                        for entry in entries:

                            if not isinstance(entry, dict):
                                continue

                            title = entry.get("title", "")

                            if "NOT RELEVANT INFORMATION" in str(title).upper():
                                continue

                            fields_are_valid = all(
                                isinstance(entry.get(field), str)
                                and entry[field].strip().upper() not in ("", "EMPTY")
                                for field in required_fields
                            )

                            if fields_are_valid:
                                valid_entries.append(entry)

                        language_entries.extend(valid_entries)
                        print(f"{batch_label} | Knowledge entries: {len(valid_entries)}")
                    except (json.JSONDecodeError, TypeError):
                        print(f"{batch_label} | Knowledge entries: unknown (invalid response)")

                knowledge_text_cleaned = json.dumps(language_entries, ensure_ascii=False)
                knowledge_output.append(knowledge_text_cleaned)
                knowledge_output_dicc[lang] = knowledge_text_cleaned
                knowledge_input.append(valid_contents)
                knowledge_input_dicc[lang] = valid_contents
                count += len(valid_contents)
                print()

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

