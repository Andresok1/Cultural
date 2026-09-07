from pathlib import Path
import json
from graphics import df_model_culture_dimension, plot_model_culture_accuracy, plot_model_culture_qtype_accuracy, plot_model_questiontype_accuracy, plot_model_culture_dimension
from studentLLM import interweb_student, openrouter_grader, openrouter_student
from datetime import datetime
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

def print_banner(text):
    width = len(text) + 10
    
    print("╭" + "─" * (width - 2) + "╮")
    print("│" + text.center(width - 2) + "│")
    print("╰" + "─" * (width - 2) + "╯")

def normalize_answer(text):
    return text.strip().lower().replace(".", "").replace(",", "")

def evaluation_result(question, llm_answer, reference_answer, question_type):
    point = 0

    if llm_answer is None:
        return "No answer"
    
    if question_type == "single_choice":
        if normalize_answer(llm_answer) == normalize_answer(reference_answer):
            point += 1
    elif question_type == "fill_the_blank":
        if normalize_answer(reference_answer) in normalize_answer(llm_answer):
            point += 1
    elif question_type == "true_false":
        if normalize_answer(llm_answer) == normalize_answer(reference_answer):
            point += 1
    elif question_type == "short_answer":
        result= openrouter_grader(question, reference_answer, llm_answer, question_type)
        if result == "PASS":
            point += 1
    elif question_type == "long_answer":
        result= openrouter_grader(question, reference_answer, llm_answer, question_type)
        if result == "PASS":
            point += 1

    return point

def examination(llm_model, examination_data):
    answers = {}
    metrics = {}
    correct_counter = 0

    print("Evaluating:", llm_model)

    failed_log = BASE_DIR / "results" / "failed_questions.log"

    for culture, dimensions in examination_data.items():

        answers[culture] = {}
        metrics[culture] = {}

        for dimension, question_types in dimensions.items():

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
                    
                    try: 
                        # answer = openrouter_student(
                        #     llm_model, 
                        #     question, 
                        #     options,
                        #     question_format
                        # )
                        answer = interweb_student(
                            llm_model, 
                            question, 
                            options,
                            question_format
                        )

                        metrics[culture][dimension][question_type]["total"] += 1    #A None answer is still being a response, so it has to be counted in the total.

                        if answer is not None: 

                            answer = answer.strip()
                            
                            point = evaluation_result(question, answer, reference_answer, question_format)

                            correct_counter += point

                            answers[culture][dimension][question_type].append({
                                "question:": question,
                                "llm_answer": answer,
                                "reference_answer": reference_answer,
                                "score" : point,
                            })   

                            metrics[culture][dimension][question_type]["correct"] += point

                    except Exception as e:

                        print(
                            f"FAILED: {llm_model} | {question[:50]}..."
                        )
                        print(e)

                        with open(
                            failed_log,
                            "a",
                            encoding="utf-8"
                        ) as log:

                            log.write(
                                "\n====================\n"
                                f"MODEL: {llm_model}\n"
                                f"CULTURE: {culture}\n"
                                f"DIMENSION: {dimension}\n"
                                f"TYPE: {question_type}\n"
                                f"QUESTION: {question}\n"
                                f"ERROR: {str(e)}\n"
                                "====================\n"
                            )

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

    return accuracy, answers, metrics


print_banner("STARTING EXAM")

question_path = BASE_DIR / "results" / "questions.json"

with open(question_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# models = [
#     "inclusionai/ling-3.0-flash-sante:free",
#     # "minimax/minimax-m3:free",
#     "qwen/qwen3.8-max-0902",
#     # "google/gemma-4-26b-a4b-it:free",
#     # "inclusionai/ling-3.0-flash-fin:free",

# ]

models = [
    "gpt-4.1-mini", 
    "gpt-4o", 
    # "llama4:16x17b", 
    # "qwen3.5:9b",

    ]

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open("results/results.txt", "a", encoding="utf-8") as f:
    f.write(f"Run started: {timestamp}\n")

examination_results = {}
results_table = []

for model in models:
    accuracy, answers, metrics = examination(model, data)

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


output_path = BASE_DIR / "results" / f"exam_results.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        examination_results,
        file,
        indent=4,
        ensure_ascii=False
    )

df_detail = pd.DataFrame(results_table)

df_detail.to_csv(
    "results/model_culture_dimension_questiontype_results.csv",
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
    "results/model_culture_questiontype_results.csv",
    index=False
)

#Graphics
plot_model_culture_accuracy(df)
plot_model_questiontype_accuracy(df)
plot_model_culture_qtype_accuracy(df)

df_dimension = df_model_culture_dimension(df_detail)

plot_model_culture_dimension(df_dimension)