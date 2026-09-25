from result_paths import EXAM_DIR, QUESTIONS_DIR
from pathlib import Path
import json
import shutil
import atexit
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from graphics import df_model_culture_dimension, plot_model_culture_accuracy, plot_model_culture_qtype_accuracy, plot_model_questiontype_accuracy, plot_model_culture_dimension
from studentLLM import interweb_student, openrouter_grader, openrouter_student,inference_student,openai_student, openai_grader
import pandas as pd


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


clear_exam_directory()
setup_examination_output()

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


semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
def evaluate_semantic_similarity(student_answer, reference_answer, threshold=0.80):
    """
    Evaluates whether a student answer is semantically similar
    to the reference answer using sentence embeddings.
    """

    student_answer = student_answer.lower().strip()
    reference_answer = reference_answer.lower().strip()

    student_embedding = semantic_model.encode([student_answer])
    reference_embedding = semantic_model.encode([reference_answer])

    similarity = cosine_similarity(student_embedding, reference_embedding)[0][0]

    return similarity >= threshold, float(similarity)


def evaluation_result(question, llm_answer, reference_answer, question_type):
    point = 0

    if llm_answer is None:
        return "No answer"
    
    if question_type == "single_choice":
        if normalize_answer(llm_answer) == normalize_answer(reference_answer):
            point += 1
    elif question_type == "fill_the_blank":
        correct, similarity =evaluate_semantic_similarity(llm_answer,reference_answer)
        if correct:
            point += 1
    elif question_type == "true_false":
        if normalize_answer(llm_answer) == normalize_answer(reference_answer):
            point += 1
    elif question_type == "short_answer":
        if GRADER_ENDPOINT == "openrouter":
            result= openrouter_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)
        if GRADER_ENDPOINT == "openai":
            result= openai_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)

        if result is None:
            raise RuntimeError("Grader returned no answer")

        if result == "PASS":
            point += 1
    elif question_type == "long_answer":
        if GRADER_ENDPOINT == "openrouter":
            result= openrouter_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)
        if GRADER_ENDPOINT == "openai":
            result= openai_grader(question, reference_answer, llm_answer, question_type, GRADER_LLM)

        if result is None:
            raise RuntimeError("Grader returned no answer")

        if result == "PASS":
            point += 1

    return point

def examination(llm_model, examination_data, student):
    answers = {}
    metrics = {}
    failed_questions = []
    correct_counter = 0

    print("\n" + "*" * 72)
    print(f"*** Evaluating student model: {llm_model} ***")
    print("*" * 72 + "\n")

    for culture, dimensions in examination_data.items():

        answers[culture] = {}
        metrics[culture] = {}

        for dimension, question_types in dimensions.items():

            print("=" * 72)
            print(f"DIMENSION: {culture} | {dimension}")
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
                        if student == "inference":
                            answer = inference_student(
                                llm_model, 
                                question, 
                                options,
                                question_format, 
                                culture,
                                dimension,
                                status_label=question_type
                            )
                        elif student == "openai":
                            answer = openai_student(
                                llm_model, 
                                question, 
                                options,
                                question_format, 
                                culture,
                                dimension
                            )
                        elif student == "openrouter":
                            answer = openrouter_student(
                                llm_model, 
                                question, 
                                options,
                                question_format, 
                            )
                        elif student == "openai":
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
                                "question:": question,
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
                                "type": question_type
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
                            "type": question_type
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


print_banner("STARTING EXAM")

question_path = QUESTIONS_DIR / "questions.json"

with open(question_path, "r", encoding="utf-8") as file:
    data = json.load(file)


STUDENT_MODELS = {
    "openai": [
        "gpt-4.1-mini",
        "gpt-4o",
    ],

    "interweb": [
        "gpt-4.1-mini",
        "gpt-4o",
    ],

    "interweb": [
        "openai/gpt-5.6-luna",
        "qwen/qwen3.7-flash",
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

STUDENT_ENDPOINT = "openai"
models = STUDENT_MODELS[STUDENT_ENDPOINT]

GRADER_ENDPOINT = "openai"
GRADER_LLM = GRADER_MODELS[GRADER_ENDPOINT]

examination_results = {}
results_table = []
failed_questions = []

for model in models:
    accuracy, answers, metrics, model_failed_questions = examination(
        model,
        data,
        STUDENT_ENDPOINT
    )

    for failed_question in model_failed_questions:
        if failed_question not in failed_questions:
            failed_questions.append(failed_question)

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


output_path = EXAM_DIR / "exam_results.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        examination_results,
        file,
        indent=4,
        ensure_ascii=False
    )

with open(EXAM_DIR / "failed_questions.json", "w", encoding="utf-8") as file:
    json.dump(
        failed_questions,
        file,
        indent=4,
        ensure_ascii=False
    )

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

#Graphics
plot_model_culture_accuracy(df)
plot_model_questiontype_accuracy(df)
plot_model_culture_qtype_accuracy(df)

df_dimension = df_model_culture_dimension(df_detail)

plot_model_culture_dimension(df_dimension)

print_examination_summary(df_detail)