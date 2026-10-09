from promptingLLM import openrouter_testing
from result_paths import RESULTS_DIR, KNOWLEDGE_DIR, QUESTIONS_DIR, EXAM_DIR
from datetime import datetime
from web_retrieval import fetch_raw_results
from utils import print_experiment_summary
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

def setup_experiment_output(clear_previous=True):
    # Remove the previous run's reports before creating the new ones.
    for pattern in ("experiment_summary_*.txt", "terminal_output_*.txt") if clear_previous else ():
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
    "--batch_size",
    type=positive_integer, 
    default=2,
    help="Documents per collective batch; all valid documents are processed."
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
    default="inference",
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
    choices=["factual", "conceptual", "misleading", "long_form", "random", "all"],
    default="all", 
    help="Which kind of question's type is created",
)

parser.add_argument(
    "--mode",
    choices=["full", "retry", "single"],
    default="full",
    help="Run the normal process, retry the dimensions in emptyKnowledge.json, or test selected cultures and dimensions.",
)

culture_language = {
    "Colombian": "spanish",
    "German": "german",
    "Italian": "italian",
}

def cultural_exam(cultures_dimensions, teacher_model=None):

    for culture, dimensions in cultures_dimensions.items():
    # for culture in cultures:
        for dimension_number, dimension in enumerate(dimensions[dimension_start:], start=dimension_start):
            print(f"Processing dimension {dimension_number+1}/{len(dimensions)}: {dimension}")
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
                summary_path=summary_path, experiment_counts=experiment_counts[culture], teacher_model=teacher_model
            )

    print("All Done!")

    for culture, counts in experiment_counts.items():
        print_experiment_summary(counts, summary_path, culture=culture)
    print(f"Summary saved to: {summary_path}")
    print(f"Terminal output saved to: {terminal_output_path}")

TEACHER_MODELS = {
    "openai": "gpt-4o",
    "interweb": "gpt-4.1-mini",
    "openrouter": "qwen/qwen3.8-max-0902",
    "inference": "gpt-oss:120b-mxfp4",
}


if __name__ == "__main__":
    args = parser.parse_args()

    culs_dims = {}

    if args.mode == "retry":
        dimension_start = 0
        # Read the failed dimensions before the result folders are cleaned.
        empty_knowledge_path = KNOWLEDGE_DIR / "emptyKnowledge.json"
        if not empty_knowledge_path.exists():
            print(f"No retry report found: {empty_knowledge_path}")
            sys.exit(0)

        with empty_knowledge_path.open("r", encoding="utf-8") as f:
            failed_dimensions = json.load(f)

        if not failed_dimensions:
            print("No failed dimensions to retry.")
            sys.exit(0)

        for item in failed_dimensions:
            culture = item["culture"]
            dimension = item["dimension"]
            if culture not in culs_dims:
                culs_dims[culture] = []
            culs_dims[culture].append(dimension)

    elif args.mode == "single":
        dimension_start = 0
        # Edit these lists to choose what to test.
        cultures = ["Colombian"]
        dimensions = [
            # "stereotypes",
            "authority"
        ]
        for culture in cultures:
            culs_dims[culture] = dimensions.copy()

    else:
        csv_path = BASE_DIR.parent / "cultural_parameters" / "cultureScope.csv"
        df = pd.read_csv(csv_path)
        dimensions = df["Fine-grained Dimension"].tolist()
        dimension_start = 0 #from 0 to 139

        cultures = [
            "Colombian",
            "German",
            "Italian",
        ]

        for cul in cultures:
            culs_dims[cul] = dimensions.copy()

    # Retry completes the saved run without deleting its results.
    if args.mode != "retry":
        for results_folder in (KNOWLEDGE_DIR, QUESTIONS_DIR, EXAM_DIR):
            for file_path in results_folder.iterdir():
                if file_path.is_file():
                    file_path.unlink()

    experiment_counts = {
        culture: {"complete": 0, "incomplete": 0}
        for culture in culs_dims.keys()
    }
    all_results = {}
    culture_dfs = {}

    if args.mode == "retry":
        query_results_path = KNOWLEDGE_DIR / "query_results.json"
        if query_results_path.exists():
            with query_results_path.open("r", encoding="utf-8") as f:
                all_results = json.load(f)
        for culture in culs_dims:
            csv_path = QUESTIONS_DIR / f"{culture}_Knowledge_QA.csv"
            if csv_path.exists():
                culture_dfs[culture] = [pd.read_csv(csv_path)]

    summary_path, terminal_output_path = setup_experiment_output(clear_previous=args.mode != "retry")

    print(f"Experiment setup ({args.mode}):")
    for culture, dimensions in culs_dims.items():
        print(f"{culture}: {len(dimensions)} dimensions")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    cultural_exam(culs_dims, teacher_model=TEACHER_MODELS[args.api])

    empty_knowledge_path = KNOWLEDGE_DIR / "emptyKnowledge.json"
    if empty_knowledge_path.exists():
        with empty_knowledge_path.open("r", encoding="utf-8") as f:
            empty_knowledge = json.load(f)
        if empty_knowledge:
            print("\nemptyKnowledge is not empty. Check results/knowledge/emptyKnowledge.json.")
