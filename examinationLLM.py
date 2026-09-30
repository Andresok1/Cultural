from result_paths import EXAM_DIR, QUESTIONS_DIR
from pathlib import Path
import json
import shutil
import atexit
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from graphics import df_model_culture_dimension
from studentLLM import interweb_student, openrouter_grader, openrouter_student,inference_student,openai_student, openai_grader,inference_grader
import pandas as pd
import argparse


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


def setup_examination_output():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = EXAM_DIR / f"exam_terminal_output_{timestamp}.txt"
    output_document = output_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    terminal_capture = TerminalCapture(sys.stdout, output_document)
    sys.stdout = terminal_capture
    sys.stderr = terminal_capture

    def close_terminal_output():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        output_document.close()

    atexit.register(close_terminal_output)

    return output_path


def clear_exam_directory():
    for item in EXAM_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def print_banner(text):
    width = len(text) + 10
    
    print("╭" + "─" * (width - 2) + "╮")
    print("│" + text.center(width - 2) + "│")
    print("╰" + "─" * (width - 2) + "╯")


def print_examination_summary(df_detail):
    culture_summary = (
        df_detail
        .groupby(["model", "culture"])
        .agg(correct=("correct", "sum"), total=("total", "sum"))
        .reset_index()
    )
    culture_summary["accuracy"] = (
        culture_summary["correct"] / culture_summary["total"]
    )

    category_summary = df_model_culture_dimension(df_detail)

    question_type_summary = (
        df_detail
        .groupby(["model", "culture", "question_type"])
        .agg(correct=("correct", "sum"), total=("total", "sum"))
        .reset_index()
    )
    question_type_summary["accuracy"] = (
        question_type_summary["correct"] /
        question_type_summary["total"]
    )

    print_banner("EXAMINATION SUMMARY")

    for model in sorted(df_detail["model"].unique()):
        print("\n" + "*" * 72)
        print(f"*** Results for student model: {model} ***")
        print("*" * 72)

        print("\nAccuracy by culture")
        print(
            culture_summary[culture_summary["model"] == model][
                ["culture", "correct", "total", "accuracy"]
            ]
            .assign(accuracy=lambda table: table["accuracy"].map("{:.2%}".format))
            .to_string(index=False)
        )

        print("\nAccuracy by category")
        print(
            category_summary[category_summary["model"] == model][
                ["culture", "category", "correct", "total", "accuracy"]
            ]
            .assign(accuracy=lambda table: table["accuracy"].map("{:.2%}".format))
            .to_string(index=False)
        )

        print("\nAccuracy by question type")
        print(
            question_type_summary[question_type_summary["model"] == model][
                ["culture", "question_type", "correct", "total", "accuracy"]
            ]
            .assign(accuracy=lambda table: table["accuracy"].map("{:.2%}".format))
            .to_string(index=False)
        )

def normalize_answer(text):
    return text.strip().lower().replace(".", "").replace(",", "")

def evaluation_result(question, llm_answer, reference_answer, question_type):
    point = 0

    if llm_answer is None:
        return "No answer"
    
    elif question_type == "single_choice":
        if normalize_answer(llm_answer) == normalize_answer(reference_answer):
            point += 1

    elif question_type == "true_false":
        if normalize_answer(llm_answer) == normalize_answer(reference_answer):
            point += 1
    
    elif question_type in ("long_answer", "short_answer", "fill_the_blank"):
        if GRADER_ENDPOINT == "openrouter":
            result= openrouter_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)
        elif GRADER_ENDPOINT == "openai":
            result= openai_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)
        elif GRADER_ENDPOINT == "inference":
            result = inference_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)
        else:
            raise ValueError(f"Unknown grader endpoint: {GRADER_ENDPOINT}")
        
        if result is None:
            raise RuntimeError("Grader returned no answer")

        if result == "PASS":
            point += 1

    return point

def load_questions():   
    question_path = QUESTIONS_DIR / "questions.json"

    with open(question_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for culture, dimensions in data.items():
        print(f"{culture}: {len(dimensions)} dimensions")
    return data

def examination(llm_model, examination_data, endpoint):
    answers = {}
    metrics = {}
    failed_questions = []
    correct_counter = 0

    for culture, dimensions in examination_data.items():

        answers[culture] = {}
        metrics[culture] = {}
        
        total_dimensions = len(dimensions)
        for idx, (dimension, question_types) in enumerate(dimensions.items(), start=1):

            print("=" * 72)
            print(f"DIMENSION: {idx}/{total_dimensions}: {culture} | {dimension}")
            print()

            answers[culture][dimension] = {}
            metrics[culture][dimension] = {}

            for question_type, questions in question_types.items():

                answers[culture][dimension][question_type] = []
                metrics[culture][dimension][question_type] = {
                    "correct": 0,
                    "total": 0
                }

                for question_data in questions:

                    question_format = question_data["format"]
                    question = question_data["question"]
                    options = question_data.get("options")
                    reference_answer = question_data["reference_answer"]

                    stage = "student"
                    try: 
                        if endpoint == "inference":
                            answer = inference_student(
                                llm_model, 
                                question, 
                                options,
                                question_format, 
                                culture,
                                dimension,
                                status_label=question_type
                            )
                        elif endpoint == "openai":
                            answer = openai_student(
                                llm_model, 
                                question, 
                                options,
                                question_format, 
                                culture,
                                dimension,
                                status_label=question_type
                            )
                        elif endpoint == "openrouter":
                            answer = openrouter_student(
                                llm_model, 
                                question, 
                                options,
                                question_format,
                                status_label=question_type
                            )
                        elif endpoint == "interweb":
                            answer = interweb_student(
                                llm_model, 
                                question, 
                                options,
                                question_format
                            )

                        metrics[culture][dimension][question_type]["total"] += 1    #A None answer is still being a response, so it has to be counted in the total.

                        if answer is not None: 

                            answer = answer.strip()

                            stage = "grader"

                            print(
                                f"{'GRADING':<9} | {question_type:<10} | "
                                f"{GRADER_ENDPOINT} / {GRADER_LLM} | "
                                "Checking answer...",
                                flush=True
                            )
                            
                            point = evaluation_result(question, answer, reference_answer, question_format)

                            print(
                                f"{'GRADING':<9} | {question_type:<10} | "
                                f"{GRADER_ENDPOINT} / {GRADER_LLM} | "
                                "Finished",
                                flush=True
                            )
                            print("\n")

                            correct_counter += point

                            answers[culture][dimension][question_type].append({
                                "question": question,
                                "llm_answer": answer,
                                "reference_answer": reference_answer,
                                "score" : point,
                            })   

                            metrics[culture][dimension][question_type]["correct"] += point

                        else:
                            failed_question = {
                                "model": llm_model,
                                "culture": culture,
                                "dimension": dimension,
                                "type": question_type,
                                "format": question_format,
                                "question": question,
                                "options": options,
                                "reference_answer": reference_answer
                            }
                            if failed_question not in failed_questions:
                                failed_questions.append(failed_question)

                    except Exception as e:
                        
                        print(
                            f"FAILED during {stage}: "
                            f"{type(e).__name__}: {e}"
                        )

                        failed_question = {
                            "model": llm_model,
                            "culture": culture,
                            "dimension": dimension,
                            "type": question_type,
                            "format": question_format,
                            "question": question,
                            "options": options,
                            "reference_answer": reference_answer
                        }
                        if failed_question not in failed_questions:
                            failed_questions.append(failed_question)

                        answer = None

    questions_size = sum(
        metrics[c][d][q]["total"]
        for c in metrics
        for d in metrics[c]
        for q in metrics[c][d]
    )

    if questions_size == 0:
        accuracy = 0
        print("Error: question_size is zero")
    else:
        accuracy = correct_counter/questions_size

    for culture in metrics:
        for dimension in metrics[culture]:
            for qtype in metrics[culture][dimension]:

                total = metrics[culture][dimension][qtype]["total"]
                correct = metrics[culture][dimension][qtype]["correct"]

                metrics[culture][dimension][qtype]["accuracy"] = (
                    correct / total if total > 0 else 0
                )

    return accuracy, answers, metrics, failed_questions




STUDENT_MODELS = {
    "openai": [
        "gpt-4.1-mini",
        "gpt-4o",
    ],

    "interweb": [
        "gpt-4.1-mini",
        "gpt-4o",
    ],

    "openrouter": [
        "qwen/qwen3.5-9b",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "cohere/north-mini-code:free",
        "liquid/lfm-2.5-2.6b:free",
    ],

    "inference": [
        "granite-4.1:8b-bf16",
        "gemma3:4b-f16",
    ],
}

GRADER_MODELS = {
    "openai": "gpt-4o",
    "interweb": "gpt-4.1-mini",
    "openrouter": "qwen/qwen3.8-max-0902",
    "inference": "granite-4.1:8b-bf16",
}

STUDENT_ENDPOINT = "openrouter"
models = STUDENT_MODELS[STUDENT_ENDPOINT]

GRADER_ENDPOINT = "inference"
GRADER_LLM = GRADER_MODELS[GRADER_ENDPOINT]

def save_exam_results(examination_results):
    output_path = EXAM_DIR / "exam_results.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            examination_results,
            file,
            indent=4,
            ensure_ascii=False
        )

def save_failed_questions(failed_questions):
    with open(EXAM_DIR / "failed_questions.json", "w", encoding="utf-8") as file:
        json.dump(
            failed_questions,
            file,
            indent=4,
            ensure_ascii=False
        )

def save_csv(results_table):
    df_detail = pd.DataFrame(results_table)

    df_detail.to_csv(
        EXAM_DIR / "model_culture_dimension_questiontype_results.csv",
        index=False
    )

    df = pd.DataFrame(results_table)

    df = (
        df
        .groupby(
            [
                "model",
                "culture",
                "question_type"
            ]
        )
        .agg(
            {
                "correct":"sum",
                "total":"sum"
            }
        )
        .reset_index()
    )

    df["accuracy"] = df["correct"] / df["total"]

    df.to_csv(
        EXAM_DIR / "model_culture_questiontype_results.csv",
        index=False
    )
    return df_detail

def run_full_exam():

    clear_exam_directory()
    setup_examination_output()

    data= load_questions()

    results_table = []
    failed_questions = []
    examination_results = {}

    for endpoint, models,  in STUDENT_MODELS.items():
        for model in models:
            print("\n" + "*" * 72)
            print(f"*** Evaluating student model: {endpoint} | {model} ***")
            print("*" * 72 + "\n")

            accuracy, answers, metrics, model_failed_questions = examination(
                model,
                data,
                endpoint
            )

            for failed_question in model_failed_questions:
                if failed_question not in failed_questions:
                    failed_questions.append(failed_question)

            print(model)

            examination_results[model] = {
                "accuracy": accuracy,
                "answers": answers
            }

            for culture in metrics:
                for dimension in metrics[culture]:
                    for qtype in metrics[culture][dimension]:

                        key = {
                            "model": model,
                            "culture": culture,
                            "dimension": dimension,
                            "question_type": qtype
                        }

                        results_table.append({
                            **key,
                            "correct": metrics[culture][dimension][qtype]["correct"],
                            "total": metrics[culture][dimension][qtype]["total"]
                        })


    save_exam_results(examination_results)
    save_failed_questions(failed_questions)
    df_detail = save_csv(results_table)
    print_examination_summary(df_detail)


def retry_failed_questions():

    with open(EXAM_DIR / "failed_questions.json") as file:
        failed_questions = json.load(file)

    remaining_failed = []

    for item in failed_questions:

        print(
            f"Retrying {item['model']} | "
            f"{item['culture']} | "
            f"{item['dimension']}"
            f"{item['type']} | "
        )

        answer = openrouter_student(
            item["model"],
            item["question"],
            item["options"],
            item["format"]
        )

        if answer is not None:

            score = evaluation_result(
                item["question"],
                answer,
                item["reference_answer"],
                item["format"]
            )

            update_exam_results(
                item,
                answer,
                score
            )
        else:
            remaining_failed.append(item)

    with open(EXAM_DIR / "failed_questions.json", "w") as file:
        json.dump(
            remaining_failed,
            file,
            indent=4
        )

def update_exam_results(item, answer, score):

    with open(EXAM_DIR / "exam_results.json") as file:
        results = json.load(file)

    model = item["model"]

    questions = results[model]["answers"]

    for culture in questions:
        if culture== item["culture"]:

            for dimension in questions[culture]:
                if dimension== item["dimension"]:
                    for qtype in questions[culture][dimension]:
                        if qtype== item["type"]:
                            print(f"exam_result.json updated ´with: {culture} | {dimension} | {qtype}")
                            questions[culture][dimension][qtype].append({
                                "question": item["question"],
                                "llm_answer": answer,
                                "reference_answer": item["reference_answer"],
                                "score": score,
                            })

    with open(EXAM_DIR / "exam_results.json", "w") as file:
        json.dump(results, file, indent=4)



def main(mode="full"):

    if mode == "full":
        print_banner("STARTING EXAM")

        run_full_exam()

    elif mode == "retry":
        print_banner("COMPLETING EXAM")
        retry_failed_questions()

    print("\nTo proceed with the charts, execute: python graphics.py")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["full", "retry"],
        default="full"
    )

    args = parser.parse_args()

    main(args.mode)
