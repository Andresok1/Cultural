from duckDuckGo import fetch_raw_results
from knowledgeGen import create_knowledge

import json


dimensions = [
    "population rank",
    # "population distribution",
    # "land area percentage",
    # "ethnicity",
    # "official languages",
    # "widely spoken languages",
    # "famous rives",
    # "climate",
    # ...
]

cultures= [
    "colombian",
    "german",
]

all_results = {}

for dimension in dimensions:
    for culture in cultures:

        query = f"{dimension} in {culture} culture"

        results = fetch_raw_results(query, fetch_full_content=True)

        all_results[query] = results


with open("results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

