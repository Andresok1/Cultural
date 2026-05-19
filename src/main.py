from duckDuckGo import fetch_raw_results
from knowledgeGen import create_knowledge
from questionGen import knowledge_to_question

import json
import pandas as pd



single= True   #single knowledge for each doc, else: one knowledge for all docs

### Search dimensions and cultures to query creation
dimensions = [
    "population rank",
    "famous rives",
    # "land area percentage",
    # "ethnicity",
    # "official languages",
    # "widely spoken languages",
    # "famous rives",
    # "climate",
    # ...
]

cultures= [
    "Colombian",
    "German",
    "Italian",
]

all_results = {}

for culture in cultures:
    for dimension in dimensions:

        query = f"{dimension} in {culture} culture"

        result = fetch_raw_results(query, fetch_full_content=True)

        all_results[query] ={

                    "dimension": dimension,
                    "culture": culture,
                    "result": result,
                }   

with open("query_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)


query_results = r"C:\Users\Andres\Repos\Cultural_thesis\query_results.json"

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

    knowledge_input= []
    knowledge_output= []

    count = 0
    max_results = 3     #TODO: args to control number of considered documents for knowledge generation

    if single:  #'''TODO: se arman grupos de knowldge aca->funcion'''
        for doc in docs:
            content = doc.get('content')   #optimizar con knowledger_input y sacar el create_knowledge del if
            if content:     
                kl= create_knowledge(text=content, culture=culture, dimension=dimension)
                knowledge_output.append(kl)    ###Knowledge result by each doc
                count += 1
            
            if count >= max_results:
                break
        if count < max_results:
            print(f"Warning: Only {count} results with content found for query '{query}' (less than the max of {max_results})")



    else: 
        for doc in docs:
            content = doc.get('content')

            if content:
                knowledge_input.append({content})
                count += 1

            if count >= max_results:
                break
        if count < max_results:
            print(f"Warning: Only {count} results with content found for query '{query}' (less than the max of {max_results})")
        
        kl= create_knowledge(text=knowledge_input, culture=culture, dimension=dimension)
        kl_cleaned = kl.replace(";", ",").strip() 
        knowledge_output.append(kl)    ###One knowledge result for all docs


    with open(f"knowledge_output_{culture}.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_output, f, ensure_ascii=False, indent=2)

    knowledge_path = rf"C:\Users\Andres\Repos\Cultural_thesis\knowledge_output_{culture}.json"

    question, abcd_options, reference_answer, knowledge_list, title_list, snippet_list = knowledge_to_question(knowledge_path)

    df_dimension = pd.DataFrame([{"culture": culture, "dimension": dimension}])

    data_knowledge_info = {}
    for i, (t, s, k) in enumerate(zip(title_list, snippet_list, knowledge_list), start=1):
        data_knowledge_info[f"title_{i}"] = [t]
        data_knowledge_info[f"snippet_{i}"] = [s]
        data_knowledge_info[f"knowledge_{i}"] = [k]

    df_knowledge_info = pd.DataFrame(data_knowledge_info)

    df_questions_reference = pd.DataFrame({
        "question": [question],
        "abcd_options": [abcd_options],
        "reference_answer": [reference_answer]
    })

    df = pd.concat([df_dimension.reset_index(drop=True),
                    df_knowledge_info.reset_index(drop=True),
                    df_questions_reference.reset_index(drop=True)], axis=1)

    culture_dfs[culture].append(df)


for culture, dfs in culture_dfs.items():
    if dfs:  
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f"{culture}_Knowledge_QA.csv", sep="\t", index=False, encoding="utf-8")
    else:
        print(f"Warning: No data to save for culture {culture}")
    