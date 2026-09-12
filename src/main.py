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
    help="Maximum documents per language used together in collective mode.",
)

parser.add_argument(
    "--question_language",
    choices=["english", "local"],
    default="english",
    help="Language in which the question will be generated.",
)

parser.add_argument(
    "--api",
    choices=["openai", "interweb", "openrouter"],
    default="interweb",
    help="API which is going to be used for knowledge and questions creation",
)

parser.add_argument(
    "--llm_model",
    choices=["gpt-4.1-mini", "gpt-4o"],
    default="gpt-4o", 
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

dimensions= random.sample(dimensions, 2)        #JUST TO TESTING
# dimensions= [
#     # "tax & accounting",
#     # "measuring system",
#     "tipical food",
# ]

timestamp = datetime.now().strftime("%m%d_%H%M")


for results_folder in (KNOWLEDGE_DIR, QUESTIONS_DIR, EXAM_DIR):
    for file_path in results_folder.iterdir():
        if file_path.is_file():
            file_path.unlink()

cultures= [
    "Colombian",
    "German",
    # "Italian",
]

culture_language = {
    "Colombian": "spanish",
    "German": "german",
    "Italian": "italian",
}

all_results = {}    

for culture in cultures:
    for dimension in dimensions:

        key = f"{culture}_{dimension}"

        base_query = f"{dimension} in {culture} culture"
        queries = {
            "english": base_query
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

with open(KNOWLEDGE_DIR / "query_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

knowledge_output= knowledge_level_manager(args, timestamp, all_results)

print("All Done!")

empty_knowledge_path = KNOWLEDGE_DIR / "emptyKnowledge.json"
if empty_knowledge_path.exists():
    with empty_knowledge_path.open("r", encoding="utf-8") as f:
        empty_knowledge = json.load(f)
    if empty_knowledge:
        print("\nemptyKnowledge is not empty. Check results/knowledge/emptyKnowledge.json.")

