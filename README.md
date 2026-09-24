# Cultural
This is a repository to create a Pipeline for Cultural understanding, with info retrieval, questions generations and evaluation.


## Setting up the virtual environment

```
conda create -n venv python=3.11
conda activate venv
pip install -r requirements.txt
```
## Test running question generation:
```
python -m src.main --knowledge_level collective --question_language english --api openrouter --question_type all
```

The terminal output is shown live and saved with a timestamp in `results/terminal_output_<timestamp>.txt`.
## Test running examination:

python -m examinationLLM  