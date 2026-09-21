import json 

def extraction_entries(text, clean):
    """Return accepted entries and whether extraction succeeded.

    An empty list or an explicit no-relevant-information result is successful.
    Missing responses, invalid JSON and malformed entries are failures.
    """
    try:
        entries = json.loads(clean(text or ""))
    except (ValueError, TypeError):
        return [], False
    
    if isinstance(entries, dict):
        entries = [entries]

    if not isinstance(entries, list):
        return [], False
    
    valid_entries = []
    valid_response = True

    for entry in entries:
        if not isinstance(entry, dict):
            valid_response = False
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
            valid_response = False

    return valid_entries, valid_response


def print_dimension_summary(culture, dimension, stats, questions):
    valid = sum(item["valid"] for item in stats.values())
    processed = sum(item["processed"] for item in stats.values())
    entries = sum(item["entries"] for item in stats.values())
    def coverage(done, total):
        return f"{100 * done / total:.1f}%" if total else "N/A"

    full_coverage = bool(stats)
    for item in stats.values():
        if item["valid"] == 0 or item["processed"] != item["valid"]:
            full_coverage = False
        if item["invalid"] > 0:
            full_coverage = False

    if questions == 4:
        all_questions_generated = True
    else: 
        all_questions_generated = False

    complete = full_coverage and entries > 0 and all_questions_generated

    print(f"\nDIMENSION FINISHED\n\nCulture: {culture}\nDimension: {dimension}")
    print(f"\nRetrieval\n   Valid documents: {valid}")
    print(f"\nKnowledge extraction\n    Processed documents: {processed}")
    print(f"    Knowledge entries: {entries}\n  Coverage: {coverage(processed, valid)}")

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

    for lang, item in stats.items():
        coverage = item['processed']/item['valid']

        if coverage >= 0.75:
            valid = True
        else: 
            valid = False
            # print(f"Threshold less than 75% for: {lang}") 


        coverage_lang[lang] = {
            "relation": f"({item['processed']}/{item['valid']})",
            "coverage": coverage,
            "valid": valid
        }

    if show:
        for lang, values in coverage_lang.items():
            print(f"    {lang}: relation={values['relation']}, coverage={values['coverage']}, valid={values['valid']}")


    return valid
