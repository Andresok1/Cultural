"""Shared result directories, resolved relative to the project."""

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
KNOWLEDGE_DIR = RESULTS_DIR / "knowledge"
QUESTIONS_DIR = RESULTS_DIR / "questions"
EXAM_DIR = RESULTS_DIR / "exam"

for directory in (KNOWLEDGE_DIR, QUESTIONS_DIR, EXAM_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Experiment runs are separate from normal results and startup cleanup.
EXPERIMENT_DIR = Path(__file__).resolve().parent / "experiment"
