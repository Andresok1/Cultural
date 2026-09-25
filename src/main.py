from promptingLLM import openrouter_testing
from questionGen import inference_test
from result_paths import RESULTS_DIR, KNOWLEDGE_DIR, QUESTIONS_DIR, EXAM_DIR
from dimension_summary import print_experiment_summary
from datetime import datetime
from duckDuckGo import fetch_raw_results
from knowledgeGen import knowledge_level_manager, translate
from pathlib import Path

import pandas as pd
import json
import argparse
import atexit
import sys

BASE_DIR = Path(__file__).resolve().parent


class TerminalCapture:
    def __init__(self, terminal, document):
        self.terminal = terminal
        self.document = document

    def write(self, text):
        self.terminal.write(text)
        self.document.write(text)
        self.document.flush()

    def flush(self):
        self.terminal.flush()
        self.document.flush()

def setup_experiment_output():
    # Remove the previous run's reports before creating the new ones.
    for pattern in ("experiment_summary_*.txt", "terminal_output_*.txt"):
        for file_path in RESULTS_DIR.glob(pattern):
            if file_path.is_file():
                file_path.unlink()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    summary_path = RESULTS_DIR / f"experiment_summary_{timestamp}.txt"
    terminal_output_path = RESULTS_DIR / f"terminal_output_{timestamp}.txt"

    summary_path.write_text("", encoding="utf-8")

    terminal_output_document = terminal_output_path.open(
        "w",
        encoding="utf-8"
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TerminalCapture(original_stdout, terminal_output_document)
    sys.stderr = TerminalCapture(original_stderr, terminal_output_document)

    def close_terminal_output():
        # Python may flush or report errors after atexit callbacks finish.
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        terminal_output_document.close()

    atexit.register(close_terminal_output)

    return summary_path, terminal_output_path


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Batch size must be greater than zero")
    return number

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--knowledge_level",
    choices=["atomic", "collective"],
    default="atomic",
    help="Each document received their own knowledge  (atomic) or all documents are considered together for one knowledge (collective).",
)

parser.add_argument(
    "--batch_size",
    type=positive_integer, 
    default=2,
    help="Documents per collective batch; all valid documents are processed."
)

parser.add_argument(
    "--max_results",
    type=int,
    default=5,
    help="Legacy atomic-mode count check; ignored in collective mode. Use --batch_size for collective batches.",
)

parser.add_argument(
    "--question_language",
    choices=["english", "local"],
    default="english",
    help="Language in which the question will be generated.",
)

parser.add_argument(
    "--api",
    choices=["openai", "interweb", "openrouter", "inference"],
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

# csv_path = BASE_DIR.parent / "cultural_parameters" / "cultureScope.csv"
# df = pd.read_csv(csv_path)
# dimensions = df["Fine-grained Dimension"].tolist()


dimensions= [
#     # "tax & accounting",
    "measuring system",
    # "meeting duration during the business meeting",
]

timestamp = datetime.now().strftime("%m%d_%H%M")

#Result folder cleaning before starting the process
for results_folder in (KNOWLEDGE_DIR, QUESTIONS_DIR, EXAM_DIR):
    for file_path in results_folder.iterdir():
        if file_path.is_file():
            file_path.unlink()

cultures= [
    "Colombian",
#     "German",
#     "Italian",
]

culture_language = {
    "Colombian": "spanish",
    "German": "german",
    "Italian": "italian",
}

all_results = {}
culture_dfs = {}
experiment_counts = {culture: {"complete": 0, "incomplete": 0} for culture in cultures}

summary_path, terminal_output_path = setup_experiment_output()

print(f"# dimensions to compute: {len(dimensions)} (starting at {dimension_start}/{len(all_dimensions)})")

for culture in cultures:
    for dimension_number, dimension in enumerate(dimensions, start=dimension_start):
        print(f"Processing dimension {dimension_number}/{len(all_dimensions)}: {dimension}")
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

        # Finish knowledge and questions before retrieving the next dimension.
        knowledge_level_manager(
            args, timestamp, {key: all_results[key]}, culture_dfs,
            summary_path=summary_path, experiment_counts=experiment_counts[culture],
        )

print("All Done!")

empty_knowledge_path = KNOWLEDGE_DIR / "emptyKnowledge.json"
if empty_knowledge_path.exists():
    with empty_knowledge_path.open("r", encoding="utf-8") as f:
        empty_knowledge = json.load(f)
    if empty_knowledge:
        print("\nemptyKnowledge is not empty. Check results/knowledge/emptyKnowledge.json.")

for culture, counts in experiment_counts.items():
    print_experiment_summary(counts, summary_path, culture=culture)
print(f"Summary saved to: {summary_path}")
print(f"Terminal output saved to: {terminal_output_path}")

