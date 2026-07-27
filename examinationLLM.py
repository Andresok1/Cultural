from pathlib import Path
import argparse
import json
from studentLLM import interweb_student

BASE_DIR = Path(__file__).resolve().parent

def print_banner(text):
    width = len(text) + 10
    
    print("╭" + "─" * (width - 2) + "╮")
    print("│" + text.center(width - 2) + "│")
    print("╰" + "─" * (width - 2) + "╯")

def examination(llm_model, examination_data):
    answers = {}
    correct_counter = 0
    questions_size = 0

    print("Evaluating:", llm_model)

    for culture, dimensions in examination_data.items():

        answers[culture] = {}

        for dimension, question_data in dimensions.items():

            question = question_data["question"]
            options = question_data["options"]

            reference_answer = question_data["reference_answer"].strip()
            questions_size += 1

            answer = interweb_student(
                llm_model, 
                question, 
                options
            )

            if answer is not None: 
                answer = answer.strip()
            
            answers[culture][dimension] = {
                "llm_answer": answer,
                "reference_answer": reference_answer
            }

            if answer == reference_answer:
                correct_counter += 1


    accuracy = correct_counter/questions_size

    print(f"{llm_model}: {correct_counter}/{questions_size} - {accuracy} ")

    return accuracy, answers

print_banner("STARTING EXAM")

question_path = BASE_DIR / "results" / "questions.json"

with open(question_path, "r", encoding="utf-8") as file:
    data = json.load(file)

models = ["gpt-4.1-mini", "gemma3:27b", "qwen3.6:35b", "llama3:70b", "deepseek-r1:32b"]
# models = ["gpt-4.1-mini", "gemma3:27b"]


examination_results = {}

for model in models:
    accuracy, answers = examination(model, data)

    examination_results[model] = {
        "accuracy": accuracy,
        "answers": answers
    }

# print(results)

output_path = BASE_DIR / "results" / f"examination.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        examination_results,
        file,
        indent=4,
        ensure_ascii=False
    )