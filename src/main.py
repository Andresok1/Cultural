from datetime import datetime
from duckDuckGo import fetch_raw_results
from knowledgeGen import knowledge_level_manager, translate
from pathlib import Path


import pandas as pd
import json
import argparse
import random
import glob
import os


BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--knowledge_level",
    choices=["atomic", "collective"],
    default="atomic",
    help="Each document received their own knowledge  (atomic) or all documents are considered together for one knowledge (collective).",
)

parser.add_argument(
    "--max_results",
    type=int,
    default=5,
    help="Maximum number of docs to consider for knowledge generation. Just for single-knowledge-level.",
)

parser.add_argument(
    "--question_language",
    choices=["english", "local"],
    default="english",
    help="Language in which the question will be generated.",
)

parser.add_argument(
    "--api",
    choices=["openai", "interweb"],
    default="interweb",
    help="API which is going to be used for knowledge and questions creation",
)

parser.add_argument(
    "--llm_model",
    choices=["gpt-4.1-mini", "gemma3:27b", "qwen3.6:35b", "llama3:70b", "deepseek-r1:32b"],
    default="gpt-4.1-mini", 
    help="which LLM model is going to be used for knowledge and questions creation",
)

parser.add_argument(
    "--question_type",
    choices=["factual", "conceptual", "misleading", "multihop", "random", "all"],
    default="factual", 
    help="Which kind of question's type is created",
)

args = parser.parse_args()

csv_path = BASE_DIR.parent / "cultural_parameters" / "cultureScope.csv"
df = pd.read_csv(csv_path)
dimensions = df["Fine-grained Dimension"].tolist()

dimensions= random.sample(dimensions, 1)        #JUST TO TESTING
# dimensions= [
#     # "tax & accounting",
#     # "measuring system",
#     "tipical food",
# ]

timestamp = datetime.now().strftime("%m%d_%H%M")


results_folder = BASE_DIR.parent / "results" 
for file_path in glob.glob(os.path.join(results_folder, "*")):
    if os.path.isfile(file_path):
        os.remove(file_path)
    
cultures= [
    "Colombian",
    # "German",
    # "Italian",
]

culture_language = {
    "Colombian": "es",
    "German": "de",
    "Italian": "it",
}

all_results = {}    

for culture in cultures:
    for dimension in dimensions:

        key = f"{culture}_{dimension}"

        base_query = f"{dimension} in {culture} culture"
        queries = {
            "en": base_query
        }

        lang = culture_language.get(culture)
        if lang:
            queries[lang] = translate(base_query, lang)

        languages = {}
        print(f"\n")
        print(f"############################################")
        for lang, query in queries.items():
            print(f"Query: {query} ")
            print(f"------------{lang}--------------")
            ranking = fetch_raw_results(query, key)

            languages[lang] = {
                "query": query,
                "ranking": ranking
            }

            print(f"********************************************")
    
        all_results[key] ={
            "culture": culture,
            "dimension": dimension,
            "languages": languages
        }   


for key, results in all_results.items():
    languagues = results.get("languages")
    
    print(f"###{key}'[# docs]:###")
    
    key_size = 0
    for lang, info in languagues.items():
        ranking = info.get("ranking")
        print(f"- {lang}: {len(ranking)} ")
        key_size += len(ranking)
    print(f"--------TOTAL: {key_size} --------")
    print("\n")

with open("results/query_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

knowledge_output= knowledge_level_manager(args, timestamp, all_results)

print("All Done!")

