import json
import os
import tempfile
from pathlib import Path
from result_paths import KNOWLEDGE_DIR

def atomic_write_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as document:
            json.dump(value, document, ensure_ascii=False, indent=2)
            document.flush()
            os.fsync(document.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def remove_dimension(path, culture, dimension):
    path = Path(path)
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
        atomic_write_json(path, [entry for entry in entries
                                if (entry["culture"], entry["dimension"]) != (culture, dimension)])


def reconstruct_queue(cultures, dimensions, questions, failures):
    failed = {(entry["culture"], entry["dimension"]) for entry in failures}
    types = ("factual", "conceptual", "misleading", "long_form")
    return [{"culture": culture, "dimension": dimension}
            for culture in cultures for dimension in dimensions
            if (culture, dimension) not in failed
            and not all(questions.get(culture, {}).get(dimension, {}).get(typ) for typ in types)]


def update_empty_knowledge_report(culture, dimension):
    report_path = KNOWLEDGE_DIR / "emptyKnowledge.json"

    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = []

    report = [
        entry for entry in report
        if (entry["culture"], entry["dimension"]) != (culture, dimension)
    ]

    report.append({
        "culture": culture,
        "dimension": dimension
    })

    atomic_write_json(report_path, report)

import json 

def extraction_entries(text, clean):
    """Return accepted entries, processability, and a response status.

    A response with at least one valid entry is processable, even if another entry is malformed. An empty list or an explicit no-relevant-information result is also successful.
    """
    try:
        entries = json.loads(clean(text or ""))
    except (ValueError, TypeError):
        return [], False, "invalid_json"
    
    if isinstance(entries, dict):
        entries = [entries]

    if not isinstance(entries, list):
        return [], False, "response_is_not_a_list"
    
    valid_entries = []
    invalid_entries = 0

    for entry in entries:
        if not isinstance(entry, dict):
            invalid_entries += 1
            continue

        title = str(entry.get("title", ""))
        if "NOT RELEVANT INFORMATION" in title.upper():
            continue

        fields_are_valid = True
        for field in ("title", "snippet", "knowledge"):
            value = entry.get(field)
            if not isinstance(value, str):
                fields_are_valid = False
                break
            if value.strip().upper() in ("", "EMPTY"):
                fields_are_valid = False
                break

        if fields_are_valid:
            valid_entries.append(entry)
        else:
            invalid_entries += 1

    if invalid_entries and valid_entries:
        return valid_entries, True, "partial"
    if invalid_entries:
        return [], False, "no_valid_entries"
    return valid_entries, True, "valid"


def print_dimension_summary(culture, dimension, stats, questions, valid_process):
    valid = sum(item["valid"] for item in stats.values())
    processed = sum(item["processed"] for item in stats.values())
    invalid = sum(item["invalid"] for item in stats.values())
    entries = sum(item["entries"] for item in stats.values())
    def general_coverage(done, total):
        return f"{100 * done / total:.1f}%" if total else "N/A"

    all_questions_generated = (questions == 4)

    complete =  valid_process and entries > 0 and all_questions_generated

    print(f"\nDIMENSION FINISHED\n\nCulture: {culture}\nDimension: {dimension}")
    print(f"\nRetrieval\n   Valid documents: {valid}")
    print(f"\nKnowledge extraction\n    Processed documents: {processed}")
    print(f"\nQuestions generation\n    Questions: {questions}")

    print(f"    Knowledge entries: {entries}\n  Coverage: {general_coverage(processed, valid)}")

    print("\nCoverage by language (processed / valid documents)")

    coverage_threshold(stats, show=True)
        
    print(f"\nQuestions\nGenerated: {'YES' if all_questions_generated else 'NO'}")
    print("\nExecution status: COMPLETED")
    print(f"Experimental status: {'COMPLETE' if complete else 'INCOMPLETE'}")
    print("===============================")
    return complete


def write_summary(text, summary_path=None):
    print(text)
    if summary_path is not None:
        with open(summary_path, "a", encoding="utf-8") as report:
            report.write(text + "\n")


def print_experiment_summary(counts, summary_path=None, culture=None):
    culture_label = f"Culture: {culture}\n" if culture is not None else ""
    write_summary(
        "\n" + "=" * 50 + "\nEXPERIMENT SUMMARY\n"
        + culture_label
        + f"Dimensions: {counts['complete'] + counts['incomplete']}\n"
        f"Complete:      {counts['complete']}\n"
        f"Incomplete:    {counts['incomplete']}",
        summary_path,
    )

def coverage_threshold(stats, show=False):
    coverage_lang = {}
    overall_valid = 0
    overall_processed= 0
    for lang, item in stats.items():

        overall_valid += item['valid']
        overall_processed += item['processed']

        if item['valid'] == 0:
            coverage=0
        else:
            coverage = item['processed']/item['valid']

        if coverage >= 0.75:
            valid = True
        else: 
            valid = False

        coverage_lang[lang] = {
            "relation": f"({item['processed']}/{item['valid']})",
            "coverage": coverage,
            "valid": valid
        }
        
    if overall_valid == 0:
        overall_coverage = 0
    else:
        overall_coverage= overall_processed/overall_valid

    valid_question = overall_coverage >= 0.75

    if show:
        for lang, values in coverage_lang.items():
            print(f"    {lang}: relation={values['relation']}, coverage={values['coverage']}, valid={values['valid']}")

    return valid_question
