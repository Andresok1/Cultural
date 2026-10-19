from result_paths import KNOWLEDGE_DIR
from utils import atomic_write_json, remove_dimension
import json
import os
import tiktoken

from contextlib import redirect_stdout
from io import StringIO
from questionGen import question_creation
from utils import update_empty_knowledge_report, extraction_entries, print_dimension_summary, coverage_threshold, write_summary
from promptingLLM import interweb_create_knowledge, json_cleanig, openai_create_knowledge, openrouter_create_knowledge, inference_create_knowledge, PROMPT_KNOWLEDGE, ROLE_KNOWLEDGE
from deep_translator import MyMemoryTranslator
from math import ceil


def inference_batch_entries(args, texts, culture, dimension, teacher_model, label):
    """Check and split the existing document batch before calling the teacher."""

    TEACHER_CONTEXT_WINDOWS = {
        "gpt-oss:120b-mxfp4": 131072,
    }

    context = TEACHER_CONTEXT_WINDOWS.get(teacher_model)

    if context is None:
        raise ValueError(
            f"Unknown context window for teacher model: {teacher_model}"
        )

    output_tokens = ceil(context * 0.10)
    args.inference_output_tokens = output_tokens
    safety_tokens = ceil(context * 0.02)

    input_limit = context - output_tokens - safety_tokens
    encoding = tiktoken.get_encoding("o200k_harmony")

    def count(parts):
        documents = "\n\n".join(f"Text {i}:\n{text}" for i, text in enumerate(parts, 1))

        return sum(
            len(encoding.encode(value, disallowed_special=()))
            for value in (
                "system",
                ROLE_KNOWLEDGE,
                "user",
                PROMPT_KNOWLEDGE(culture, dimension, documents) + "\n"
            )
        )

    pending = [(texts, 0)]
    findings = []
    failure_status = None

    while pending:
        parts, depth = pending.pop(0)
        overhead = count([""] * len(parts))
        tokens = count(parts)

        print(f"{label} |"
              f"remaining tokens={input_limit - tokens}")

        if overhead >= input_limit:
            raise ValueError("Extraction instructions leave no document budget")
        
        too_large = tokens > input_limit
        response = False if too_large else inference_create_knowledge(
            args, text=parts, culture=culture, dimension=dimension,
            language=label, teacher_model=teacher_model)
        if response is False:
            if (too_large and len(parts) == 1 and len(parts[0]) <= 1) or (
                    not too_large and (depth >= 12 or (len(parts) == 1 and len(parts[0]) <= 256))):
                failure_status = "unresolved_token_limit"
                continue
            if len(parts) > 1:
                middle = len(parts) // 2
                children = [parts[:middle], parts[middle:]]
            else:
                middle = len(parts[0]) // 2
                boundary = parts[0].rfind(" ", middle // 2, middle)
                if boundary > 0:
                    middle = boundary + 1
                children = [[parts[0][:middle]], [parts[0][middle:]]]
            pending[:0] = [(child, depth + (not too_large)) for child in children]
            print(f"{label} | Splitting input to fit token limits")
            continue
        entries, valid, status = extraction_entries(response, json_cleanig)
        if not valid:
            failure_status = status
            continue
        findings.extend(entries)
    return findings, failure_status is None, failure_status or "valid"

def translate(text, target_lang):
    try:
        answer= MyMemoryTranslator(
            source="english",
            target=target_lang
        ).translate(text) 
        # print(answer)
        return answer
    except Exception as e:
        print(e)
        return text


def knowledge_level_manager(args, timestamp, query_results, culture_dfs=None,
                            summary_path=None, experiment_counts=None, teacher_model=None):
    """
    It manages between atomic and collective to organize knowledge generation.
    Pass a shared culture_dfs to retain CSV rows across dimension-level calls.
    """

    if culture_dfs is None:
        culture_dfs = {}
    
    for key, info in query_results.items():
        culture = info.get('culture', [])
        dimension = info.get('dimension')
        languages = info.get('languages', {})
        stats = {}
        for lang, data in languages.items():
            valid_documents = 0
            for document in data.get("ranking", []):
                if isinstance(document, dict):
                    content = document.get("text")
                else:
                    content = document

                if content:
                    valid_documents += 1

            stats[lang] = {
                "valid": valid_documents,
                "processed": 0,
                "entries": 0,
                "invalid": 0,
            }

        if culture not in culture_dfs:
            culture_dfs[culture] = []

        output_path = KNOWLEDGE_DIR / "knowledge_output_raw.json"
        input_path = KNOWLEDGE_DIR / "knowledge_input.json"

        if os.path.exists(output_path): #Update for knowledge_output
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    knowledge_output_dict = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {output_path} is invalid JSON. Reinitializing.")
                knowledge_output_dict = {}
        else:
            knowledge_output_dict = {}     #Structure in Knowledge_output.json

        if os.path.exists(input_path):  #Update for knowledge_input
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    knowledge_input_dict = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {input_path} is invalid JSON. Reinitializing.")
                knowledge_input_dict = {}
        else:
            knowledge_input_dict = {}

        knowledge_output= []    #For each Query
        knowledge_input= []
        knowledge_input_dicc = {}
        knowledge_output_dicc = {}
        count = 0
        
        print(f"###{key}'[# docs]:###")
        for lang, data in languages.items():
            print(f"- {lang}: {len(data.get('ranking', []))}")
        total = sum(len(data.get("ranking", [])) for data in languages.values())
        print(f"--------TOTAL: {total} --------\n")

# collective:
        print(f"Knowledge | {culture} | {dimension}")
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for lang, data in languages.items():
            ranking = data.get("ranking", [])
            valid_contents = []
            for document in ranking:
                if isinstance(document, dict):
                    content = document.get("text")
                else:
                    content = document

                if content:
                    valid_contents.append(document)
            language_entries = []
            total_batches = (len(valid_contents) + args.batch_size - 1) // args.batch_size  #upper bound division for total number of batches

            for offset in range(0, len(valid_contents), args.batch_size):
                knowledge_input_cache = valid_contents[offset:offset + args.batch_size] #batch slider
                prompt_texts = [document["text"] if isinstance(document, dict) else document
                                for document in knowledge_input_cache]
                batch_label = f"{lang} | Batch {offset // args.batch_size + 1}/{total_batches} | docs: {len(knowledge_input_cache)}"

                if args.api == "openai":
                    knowledge_text = openai_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label, teacher_model=teacher_model)
                elif args.api == "openrouter":
                    knowledge_text = openrouter_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label, teacher_model=teacher_model)
                elif args.api == "inference":
                    valid_entries, valid_response, response_status = inference_batch_entries(
                        args, prompt_texts, culture, dimension, teacher_model, batch_label)
                else:
                    knowledge_text = interweb_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label, teacher_model=teacher_model)

                if args.api != "inference":
                    valid_entries, valid_response, response_status = extraction_entries(
                        knowledge_text,
                        json_cleanig,
                    )

                language_entries.extend(valid_entries)
                
                stats[lang]["entries"] += len(valid_entries)

                if valid_response:
                    stats[lang]["processed"] += len(knowledge_input_cache)
                    print(
                        f"{batch_label} | Status: {response_status} | "
                        f"Knowledge entries: {len(valid_entries)}"
                    )
                else:
                    stats[lang]["invalid"] += 1
                    print(
                        f"{batch_label} | Invalid response: {response_status}; "
                        "documents not counted as processed"
                    )

            knowledge_text_cleaned = json.dumps(language_entries, ensure_ascii=False)
            knowledge_output.append(knowledge_text_cleaned)
            knowledge_output_dicc[lang] = knowledge_text_cleaned
            knowledge_input.append(valid_contents)
            knowledge_input_dicc[lang] = valid_contents
            count += len(valid_contents)
            print()

        if culture not in knowledge_output_dict:
            knowledge_output_dict[culture] = {}

        if culture not in knowledge_input_dict:
            knowledge_input_dict[culture] = {}

 
        knowledge_output_dict[culture][dimension] = knowledge_output_dicc

        knowledge_input_dict[culture][dimension] = knowledge_input_dicc


        atomic_write_json(output_path, knowledge_output_dict)

        atomic_write_json(input_path, knowledge_input_dict)

        question_counts = None

        valid_process = coverage_threshold(stats)
        if valid_process == True:
            question_counts = question_creation(args, dimension, culture, timestamp, culture_dfs, knowledge_output_dict, teacher_model=teacher_model)
        else:
            update_empty_knowledge_report(culture, dimension)

        if summary_path is None:
            complete = print_dimension_summary(culture, dimension, stats, question_counts, valid_process)
        else:
            summary_output = StringIO()
            with redirect_stdout(summary_output):
                complete = print_dimension_summary(culture, dimension, stats, question_counts, valid_process)
            write_summary(summary_output.getvalue().rstrip("\n"), summary_path)

        if not complete:
            update_empty_knowledge_report(culture, dimension)
        
        if experiment_counts is not None:
            experiment_counts["complete" if complete else "incomplete"] += 1

        if getattr(args, "mode", None) in ("retry", "resume") and complete:
            report_path = KNOWLEDGE_DIR / "emptyKnowledge.json"
            remove_dimension(report_path, culture, dimension)


    return knowledge_output

