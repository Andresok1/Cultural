from result_paths import EXAM_DIR, QUESTIONS_DIR
from pathlib import Path
import json
import shutil
import atexit
import sys
import time
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

    assessment_type_summary = (
        df_detail
        .groupby(["model", "culture", "assessment_type"])
        .agg(correct=("correct", "sum"), total=("total", "sum"))
        .reset_index()
    )
    assessment_type_summary["accuracy"] = (
        assessment_type_summary["correct"] /
        assessment_type_summary["total"]
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

        print("\nAccuracy by assessment type")
        print(
            assessment_type_summary[assessment_type_summary["model"] == model][
                ["culture", "assessment_type", "correct", "total", "accuracy"]
            ]
            .assign(accuracy=lambda table: table["accuracy"].map("{:.2%}".format))
            .to_string(index=False)
        )

def normalize_answer(text):
    return text.strip().lower().replace(".", "").replace(",", "")

def evaluation_result(question, llm_answer, reference_answer, question_type):
    point = 0

    print(
    f"{'GRADING':<9} | {question_type:<10} | "
    f"{GRADER_ENDPOINT} / {GRADER_LLM} | "
    "Checking answer...",
    flush=True
    )

    if not isinstance(llm_answer, str) or not llm_answer.strip():
        raise RuntimeError("Student returned no answer")
    
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
        
        if not isinstance(result, str) or result.strip() not in ("PASS", "FAIL"):
            raise RuntimeError("Grader returned no valid score")

        result = result.strip()

        if result == "PASS":
            point += 1

    
    else:
        raise ValueError(f"Unsupported question format: {question_type}")

    print(
        f"{'GRADING':<9} | {question_type:<10} | "
        f"{GRADER_ENDPOINT} / {GRADER_LLM} | "
        "Finished",
        flush=True
    )
    print("\n")

    return point

def load_questions():
    question_path = QUESTIONS_DIR / "questions.json"

    with open(question_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for culture, dimensions in data.items():
        print(f"{culture}: {len(dimensions)} dimensions")
    return data


def validate_question_dataset(data):
    """Infer expected types from the dimension with the most populated types."""
    dimensions = [
        (culture, dimension, types)
        for culture, culture_dimensions in data.items()
        for dimension, types in culture_dimensions.items()
    ]
    question_types = sorted({
        qtype for _, _, types in dimensions for qtype in types
    })
    print("\nQUESTION COUNTS BEFORE EXAMINATION")
    print(" | ".join(["Culture", "Dimensions", *question_types, "Total"]))
    for culture, culture_dimensions in data.items():
        counts = {
            qtype: sum(len(types.get(qtype, []))
                       for types in culture_dimensions.values())
            for qtype in question_types
        }
        print(" | ".join([culture, str(len(culture_dimensions)),
                          *(str(counts[qtype]) for qtype in question_types),
                          str(sum(counts.values()))]))

    if not dimensions:
        print("Question dataset is empty.")
        return False

    culture, dimension, reference = max(
        dimensions, key=lambda item: sum(bool(questions) for questions in item[2].values())
    )
    expected = {qtype for qtype, questions in reference.items() if questions}
    print(f"\nReference dimension: {culture} | {dimension}")
    print(f"Expected question types: {', '.join(sorted(expected)) or 'none'}")
    valid = bool(expected)
    for culture, dimension, types in dimensions:
        present = {qtype for qtype, questions in types.items() if questions}
        missing = expected - present
        unexpected = present - expected
        if missing:
            print(f"{culture} | {dimension} | Missing: {', '.join(sorted(missing))}")
            valid = False
        if unexpected:
            print(f"{culture} | {dimension} | Unexpected types: {', '.join(sorted(unexpected))}")
            valid = False
    if not valid:
        print("Question dataset incomplete or inconsistent. Complete question generation before examination.")
    return valid


def student_answer(endpoint, model, question, options, question_format,
                   culture, dimension, question_type):
    if endpoint == "inference":
        return inference_student(model, question, options, question_format,
                                 culture, dimension, status_label=question_type)
    if endpoint == "openai":
        return openai_student(model, question, options, question_format,
                              culture, dimension, status_label=question_type)
    if endpoint == "openrouter":
        return openrouter_student(model, question, options, question_format,
                                 status_label=question_type)
    if endpoint == "interweb":
        return interweb_student(model, question, options, question_format)
    raise ValueError(f"Unknown student endpoint: {endpoint}")


def examination_complete(results, data):
    """Require exactly one answered, scored response to every expected question."""
    complete = bool(results) and bool(data)
    run_path = EXAM_DIR / "exam_run.json"
    if run_path.exists():
        expected_models = set(json.loads(run_path.read_text(encoding="utf-8"))["model_endpoints"])
    else:
        expected_models = {model for models in STUDENT_MODELS.values() for model in models}
    if expected_models - results.keys():
        print(f"Missing student models: {sorted(expected_models - results.keys())}")
        complete = False
    expected = {
        (culture, dimension, qtype, item["question"], item["reference_answer"])
        for culture, dimensions in data.items()
        for dimension, types in dimensions.items()
        for qtype, items in types.items()
        for item in items
    }
    for model, result in results.items():
        actual = []
        for culture, dimensions in result["answers"].items():
            for dimension, types in dimensions.items():
                for qtype, answers in types.items():
                    for answer in answers:
                        if (answer.get("score") not in (0, 1)
                                or not isinstance(answer.get("llm_answer"), str)
                                or not answer["llm_answer"].strip()):
                            complete = False
                        actual.append((culture, dimension, qtype, answer["question"],
                                       answer["reference_answer"]))
        if set(actual) != expected or len(actual) != len(expected):
            print(f"{model}: incomplete question set or duplicate responses")
            complete = False
    if not complete:
        print("Examination incomplete. Retry pending questions before comparing models.")
    return complete

def examination(llm_model, examination_data, endpoint, checkpoint=None):
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

                    try:
                        answer = student_answer(
                            endpoint, llm_model, question, options, question_format,
                            culture, dimension, question_type
                        )

                        if isinstance(answer, str) and answer.strip(): 

                            answer = answer.strip()


                           
                            point = evaluation_result(question, answer, reference_answer, question_format)

                            metrics[culture][dimension][question_type]["total"] += 1
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
                                "endpoint": endpoint,
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
                        print(f"FAILED: {type(e).__name__}: {e}")
                        failed_question = {
                            "model": llm_model,
                            "endpoint": endpoint,
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

                    if checkpoint is not None:
                        checkpoint(llm_model, answers, failed_questions)

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
    # "openai": [
    #     "gpt-4.1-mini",
    #     "gpt-4o",
    # ],

    # "interweb": [
    #     "gpt-4.1-mini",
    #     "gpt-4o",
    # ],

    # "openrouter": [
    #     "qwen/qwen3.5-9b",
        # "nvidia/nemotron-3-ultra-550b-a55b:free",
        # "cohere/north-mini-code:free",
        # "liquid/lfm-2.5-2.6b:free",
    # ],

    "inference": [
        "granite-4.1:8b-bf16",
        "gemma3:4b-f16",
        "apertus:8b-2509-it-bf16",
        "qwen3-vl:8b",
    ],
}


GRADER_MODELS = {
    "openai": "gpt-4o",
    "interweb": "gpt-4.1-mini",
    "openrouter": "qwen/qwen3.8-max-0902",
    "inference": "gpt-oss:120b-mxfp4",
}

GRADER_ENDPOINT = "inference"
GRADER_LLM = GRADER_MODELS[GRADER_ENDPOINT]

def save_exam_json(filename, value):
    """Replace a checkpoint only after its complete JSON has been written."""
    output_path = EXAM_DIR / filename
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=4, ensure_ascii=False), encoding="utf-8")
    # Windows may temporarily lock the destination while another process reads it.
    for attempt in range(6):  # Initial attempt plus five retries.
        try:
            temporary_path.replace(output_path)
            return
        except PermissionError as error:
            if attempt == 5:
                raise PermissionError(
                    f"Could not replace {output_path} after five retries. "
                    f"The temporary checkpoint remains at {temporary_path}. "
                    "Resolve the file lock or permissions, then resume with "
                    "python -m examinationLLM --mode retry."
                ) from error
            time.sleep(0.5)


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
                "assessment_type"
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

    data = load_questions()
    if not validate_question_dataset(data):
        raise SystemExit("Examination stopped; existing examination results were preserved.")

    clear_exam_directory()
    setup_examination_output()

    results_table = []
    failed_questions = []
    model_endpoints = {model: endpoint for endpoint, models in STUDENT_MODELS.items()
                       for model in models}
    examination_results = {model: {"accuracy": 0, "answers": {}}
                           for model in model_endpoints}
    save_exam_json("exam_run.json", {"model_endpoints": model_endpoints})
    save_exam_json("exam_results.json", examination_results)
    save_exam_json("failed_questions.json", [])

    def checkpoint(model, answers, pending):
        """Save the current model's progress and questions needing a retry."""
        total_score = 0
        answer_count = 0

        for culture_answers in answers.values():
            for dimension_answers in culture_answers.values():
                for question_type_answers in dimension_answers.values():
                    for answer in question_type_answers:
                        total_score += answer["score"]
                        answer_count += 1

        accuracy = 0
        if answer_count > 0:
            accuracy = total_score / answer_count

        examination_results[model] = {
            "accuracy": accuracy,
            "answers": answers,
        }
        save_exam_json("exam_results.json", examination_results)

        questions_to_retry = failed_questions + pending
        save_exam_json("failed_questions.json", questions_to_retry)

    for endpoint, models,  in STUDENT_MODELS.items():
        for model in models:
            print("\n" + "*" * 72)
            print(f"*** Evaluating student model: {endpoint} | {model} ***")
            print("*" * 72 + "\n")

            accuracy, answers, metrics, model_failed_questions = examination(
                model,
                data,
                endpoint,
                checkpoint=checkpoint,
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
                            "assessment_type": qtype
                        }

                        results_table.append({
                            **key,
                            "correct": metrics[culture][dimension][qtype]["correct"],
                            "total": metrics[culture][dimension][qtype]["total"]
                        })


    save_exam_json("exam_results.json", examination_results)
    save_exam_json("failed_questions.json", failed_questions)
    df_detail = save_csv(results_table)
    if examination_complete(examination_results, data):
        print_examination_summary(df_detail)


def recover_pending_questions():
    """Recover failures and unattempted questions from the saved run's answers."""
    with open(EXAM_DIR / "exam_results.json", encoding="utf-8") as file:
        results = json.load(file)
    data = load_questions()
    run_path = EXAM_DIR / "exam_run.json"
    if run_path.exists():
        endpoints = json.loads(run_path.read_text(encoding="utf-8"))["model_endpoints"]
    else:
        # Older runs lack metadata; recover only models present in their results.
        endpoints = {}
        for model in results:
            matches = [endpoint for endpoint, models in STUDENT_MODELS.items() if model in models]
            endpoints[model] = matches[0] if len(matches) == 1 else None
    pending = []
    for model, endpoint in endpoints.items():
        result = results.setdefault(model, {"accuracy": 0, "answers": {}})
        for culture, dimensions in data.items():
            for dimension, types in dimensions.items():
                for qtype, questions in types.items():
                    saved = result["answers"].get(culture, {}).get(dimension, {}).get(qtype, [])
                    for question in questions:
                        completed = any(
                            answer.get("question") == question["question"]
                            and answer.get("reference_answer") == question["reference_answer"]
                            and answer.get("score") in (0, 1)
                            and isinstance(answer.get("llm_answer"), str)
                            and answer["llm_answer"].strip()
                            for answer in saved
                        )
                        if not completed:
                            pending.append({
                                "model": model, "endpoint": endpoint,
                                "culture": culture, "dimension": dimension,
                                "type": qtype, "format": question["format"],
                                "question": question["question"],
                                "options": question.get("options"),
                                "reference_answer": question["reference_answer"],
                            })
    save_exam_json("exam_results.json", results)
    save_exam_json("failed_questions.json", pending)
    print(f"Pending questions recovered: {len(pending)}")
    return pending


def retry_failed_questions():

    failed_questions = recover_pending_questions()

    remaining_failed = []

    for idx, item in enumerate(failed_questions):
        print(f"Failed question {idx + 1} of {len(failed_questions)}")
        print(
            f"Retrying {item['model']} | "
            f"{item['culture']} | "
            f"{item['dimension']} |"
            f"{item['type']}"
        )

        try:
            endpoint = item.get("endpoint")
            if endpoint is None:
                # Compatibility with retry entries saved before endpoints were recorded.
                endpoints = [name for name, models in STUDENT_MODELS.items()
                             if item["model"] in models]
                if len(endpoints) != 1:
                    raise ValueError("Cannot determine original endpoint for this model")
                endpoint = endpoints[0]
            answer = student_answer(
                endpoint, item["model"], item["question"], item["options"],
                item["format"], item["culture"], item["dimension"], item["type"]
            )
            score = evaluation_result(
                item["question"], answer, item["reference_answer"], item["format"]
            )
            update_exam_results(item, answer.strip(), score)
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            remaining_failed.append(item)

        # Persist progress after each item so completed retries stay completed.
        save_exam_json("failed_questions.json", remaining_failed + failed_questions[idx + 1:])

    save_exam_json("failed_questions.json", remaining_failed)
    rebuild_metrics()


def update_exam_results(failed_question, answer, score):

    with open(EXAM_DIR / "exam_results.json", "r", encoding="utf-8") as file:
        results = json.load(file)

    model = failed_question["model"]

    if score not in (0, 1) or not answer.strip():
        raise ValueError("Cannot save an unanswered or unscored question")
    questions = results[model]["answers"]
    group = questions.setdefault(failed_question["culture"], {}).setdefault(
        failed_question["dimension"], {}).setdefault(failed_question["type"], [])
    entry = {
        "question": failed_question["question"],
        "llm_answer": answer,
        "reference_answer": failed_question["reference_answer"],
        "score": score,
    }
    # Replace an existing entry if a retry was interrupted after saving its answer.
    matches = [i for i, saved in enumerate(group)
               if saved["question"] == entry["question"]
               and saved["reference_answer"] == entry["reference_answer"]]
    if len(matches) > 1:
        raise ValueError("Duplicate saved responses for retry question")
    if matches:
        group[matches[0]] = entry
    else:
        group.append(entry)
    save_exam_json("exam_results.json", results)



def rebuild_metrics():
    """Rebuild CSVs and overall accuracies from saved answers, without API calls."""
    with open(EXAM_DIR / "exam_results.json", encoding="utf-8") as file:
        results = json.load(file)

    results_table = []
    model_totals = {}
    for model, result in results.items():
        model_correct = model_total = 0
        for culture, dimensions in result["answers"].items():
            for dimension, question_types in dimensions.items():
                for qtype, answers in question_types.items():
                    questions = [answer["question"] for answer in answers]
                    if len(questions) != len(set(questions)):
                        raise ValueError(
                            f"Duplicate questions: {model} / {culture} / {dimension} / {qtype}"
                        )
                    scores = [answer["score"] for answer in answers]
                    if any(score not in (0, 1) for score in scores):
                        raise ValueError(
                            f"Invalid scores: {model} / {culture} / {dimension} / {qtype}"
                        )
                    correct, total = sum(scores), len(answers)
                    results_table.append({
                        "model": model,
                        "culture": culture,
                        "dimension": dimension,
                        "assessment_type": qtype,
                        "correct": correct,
                        "total": total,
                    })
                    model_correct += correct
                    model_total += total
        result["accuracy"] = model_correct / model_total if model_total else 0
        model_totals[model] = model_total

    if not results_table:
        raise ValueError("No answer groups found; metrics were not changed.")
    df_detail = save_csv(results_table)
    save_exam_json("exam_results.json", results)
    if examination_complete(results, load_questions()):
        print_examination_summary(df_detail)
    return model_totals


def main(mode="full"):

    if mode == "full":
        print_banner("STARTING EXAM")

        run_full_exam()

    elif mode == "retry":
        print_banner("COMPLETING EXAM")
        setup_examination_output()
        retry_failed_questions()

    elif mode == "rebuild_metrics":
        setup_examination_output()
        for model, total in rebuild_metrics().items():
            print(f"{model}: {total} saved answers")

    with open(EXAM_DIR / "exam_results.json", encoding="utf-8") as file:
        results = json.load(file)
    with open(QUESTIONS_DIR / "questions.json", encoding="utf-8") as file:
        data = json.load(file)
    if examination_complete(results, data):
        print("\nTo proceed with the charts, execute: python graphics.py")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["full", "retry", "rebuild_metrics"],
        default="full"
    )

    args = parser.parse_args()

    main(args.mode)
