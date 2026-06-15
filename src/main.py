from datetime import datetime
from duckDuckGo import fetch_raw_results
from knowledgeGen import knowledge_level_manager
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
    default=3,
    help="Maximum number of docs to consider for knowledge generation. Just for single-knowledge-level.",
)

parser.add_argument(
    "--question_language",
    choices=["english", "local"],
    default="english",
    help="Language in which the question will be generated.",
)

args = parser.parse_args()

csv_path = BASE_DIR.parent / "cultural_parameters" / "cultureScope.csv"
# df = pd.read_csv(csv_path)
# dimensions = df["Fine-grained Dimension"].tolist()

# dimensions= random.sample(dimensions, 2)        #JUST TO TESTING
dimensions= [
    "tax & accounting",
    "measurement system",
]

timestamp = datetime.now().strftime("%m%d_%H%M")


results_folder = BASE_DIR.parent / "results" 
for file_path in glob.glob(os.path.join(results_folder, "*")):
    if os.path.isfile(file_path):
        os.remove(file_path)
    
cultures= [
    "Colombia",
    # "German",
    "Italy",
]

all_results = {}

for culture in cultures:
    for dimension in dimensions:

        query = f"{dimension} in {culture}"

        ranking = fetch_raw_results(query)
        

        all_results[query] ={

                    "dimension": dimension,
                    "culture": culture,
                    "result": ranking,
                }   

with open("results/query_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

query_results = r"C:\Users\Andres\Repos\Cultural_thesis\results\query_results.json"

knowledge_output= knowledge_level_manager(args, timestamp, query_results)

print("All Done!")

