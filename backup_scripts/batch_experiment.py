"""Compare batch sizes against one saved retrieval snapshot; no question generation."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from time import perf_counter
from promptingLLM import (PROMPT_KNOWLEDGE, ROLE_KNOWLEDGE, json_cleanig,
                          openai_create_knowledge, openrouter_create_knowledge, interweb_create_knowledge)
from result_paths import EXPERIMENT_DIR, KNOWLEDGE_DIR


def run_batches(args, culture, dimension, languages):
    """Use the same provider calls, batch selection and filtering as knowledgeGen."""
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inputs = {}
    outputs = {}
    batches = []
    findings = []
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
        inputs[lang] = valid_contents
        total_batches = (len(valid_contents) + args.batch_size - 1) // args.batch_size  #upper bound division for total number of batches

        for offset in range(0, len(valid_contents), args.batch_size):
            knowledge_input_cache = valid_contents[offset:offset + args.batch_size] #batch slider
            prompt_texts = []
            document_ids = []
            for document in knowledge_input_cache:
                if isinstance(document, dict):
                    prompt_texts.append(document["text"])
                    document_ids.append(document.get("document_id"))
                else:
                    prompt_texts.append(document)
                    document_ids.append(None)
            batch_label = f"{lang} | Batch {offset // args.batch_size + 1}/{total_batches} | docs: {len(knowledge_input_cache)}"

            started = perf_counter()
            if args.api == "openai":
                knowledge_text = openai_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label)
            elif args.api == "openrouter":
                knowledge_text = openrouter_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label)
            else:
                knowledge_text = interweb_create_knowledge(args, text=prompt_texts, culture=culture, dimension=dimension, language=batch_label)

            batch = {"batch_id": batch_label, "language": lang, "document_ids": document_ids,
                     "elapsed_seconds": perf_counter() - started, "entries": [],
                     "response": {"text": knowledge_text, "attempts": []}}
            knowledge_text_cleaned = json_cleanig(knowledge_text or "")
            try:
                entries = json.loads(knowledge_text_cleaned) if knowledge_text_cleaned else []

                if isinstance(entries, dict):
                    entries = [entries]

                valid_entries = []

                required_fields = ("title", "snippet", "knowledge")

                for entry in entries:

                    if not isinstance(entry, dict):
                        continue

                    title = entry.get("title", "")

                    if "NOT RELEVANT INFORMATION" in str(title).upper():
                        continue

                    fields_are_valid = all(
                        isinstance(entry.get(field), str)
                        and entry[field].strip().upper() not in ("", "EMPTY")
                        for field in required_fields
                    )

                    if fields_are_valid:
                        valid_entries.append(entry)

                batch["entries"] = valid_entries
                language_entries.extend(valid_entries)
                for entry in valid_entries:
                    findings.append({"entry": entry, "provenance": [{"batch_id": batch_label, "document_ids": document_ids}]})
                print(f"{batch_label} | Knowledge entries: {len(valid_entries)}")
            except (json.JSONDecodeError, TypeError) as error:
                batch["parse_error"] = str(error)
                print(f"{batch_label} | Knowledge entries: unknown (invalid response)")

            batches.append(batch)
        outputs[lang] = language_entries
        print()
    unique_entries = []
    for finding in findings:
        entry = finding["entry"]
        identity = (entry["title"], entry["snippet"], entry["knowledge"])
        if identity not in unique_entries:
            unique_entries.append(identity)
    return {"inputs": inputs, "outputs": outputs, "batches": batches, "findings": findings,
            "raw_entries": len(findings), "unique_entries": len(unique_entries)}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Batch sizes must be positive")
    return number


def experiment(inputs, sizes, api, model, root=EXPERIMENT_DIR, runner=run_batches):
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Provide positive batch sizes")
    run_dir = Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "inputs.json", inputs)
    sources = KNOWLEDGE_DIR / "document_sources.json"
    if sources.exists():
        save_json(run_dir / "document_sources.json", json.loads(sources.read_text(encoding="utf-8")))
    save_json(run_dir / "settings.json", {
        "batch_sizes": sizes, "api": api, "requested_model": model,
        "effective_model": "gpt-4o-mini" if api == "openai" else model,
        "role": ROLE_KNOWLEDGE,
        "prompt_template": PROMPT_KNOWLEDGE("{culture}", "{dimension}", "{documents}"),
        "deduplication": "All findings retained, matching knowledgeGen; exact unique count for comparison only",
        "metadata_note": "Existing provider functions return text only. Tokens, retry counts and completion reasons are unavailable.",
        "review_note": "Document IDs identify batch inputs, not verified snippet sources.",
    })
    summary = []
    for size in dict.fromkeys(sizes):
        folder = run_dir / f"batch_size_{size}"
        folder.mkdir()
        results = {}
        with (folder / "review.csv").open("w", encoding="utf-8-sig", newline="") as file:
            columns = ["finding_id", "culture", "dimension", "title", "snippet", "knowledge",
                       "batch_provenance", "supported", "relevant", "duplicate_of"]
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for key, info in inputs.items():
                args = SimpleNamespace(batch_size=size, api=api, llm_model=model)
                result = runner(args, info["culture"], info["dimension"], info["languages"])
                results[key] = result
                save_json(folder / "batches.json", results)
                for number, finding in enumerate(result["findings"], 1):
                    writer.writerow({"finding_id": f"{key}:{number}", "culture": info["culture"],
                                     "dimension": info["dimension"],
                                     **{field: finding["entry"][field] for field in ("title", "snippet", "knowledge")},
                                     "batch_provenance": json.dumps(finding["provenance"], ensure_ascii=False)})
        batches = [batch for result in results.values() for batch in result["batches"]]
        attempts = [attempt for batch in batches for attempt in batch["response"].get("attempts", [])]
        usage_complete = bool(attempts) and all(
            isinstance(attempt.get("usage"), dict) and isinstance(attempt["usage"].get("total_tokens"), int)
            for attempt in attempts)
        summary.append({
            "batch_size": size, "batches": len(batches), "provider_calls": len(batches), "request_attempts": None,
            "elapsed_seconds": sum(batch["elapsed_seconds"] for batch in batches),
            "raw_entries": sum(result["raw_entries"] for result in results.values()),
            "unique_entries": sum(result["unique_entries"] for result in results.values()),
            "total_tokens": sum(attempt["usage"]["total_tokens"] for attempt in attempts) if usage_complete else None,
            "token_usage_complete": usage_complete,
            "truncated_attempts": None,
            "failed_attempts": None,
            "invalid_batches": sum("parse_error" in batch for batch in batches),
        })
        save_json(run_dir / "summary.json", summary)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=KNOWLEDGE_DIR / "query_results.json")
    parser.add_argument("--batch_sizes", nargs="+", type=positive_integer, default=[2, 3, 5])
    parser.add_argument("--api", choices=["interweb", "openai", "openrouter"], default="interweb")
    parser.add_argument("--llm_model", default="gpt-4o")
    args = parser.parse_args()
    inputs = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"Experiment saved: {experiment(inputs, args.batch_sizes, args.api, args.llm_model)}")


if __name__ == "__main__":
    main()
