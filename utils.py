import json
from result_paths import KNOWLEDGE_DIR

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

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)