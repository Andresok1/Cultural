# Cultural thesis

A pipeline for retrieving cultural information, generating knowledge and questions,
and evaluating language models against reference answers. It covers Colombian,
German, and Italian cultures using the dimensions in
`cultural_parameters/taxonomy_dimensions.csv`.

## Setup

Run commands from the repository root in your Python environment:

```powershell
conda create -n cultural python=3.11
conda activate cultural
pip install -r requirements.txt
```

Create a `.env` file with the credentials required by your selected providers:

```dotenv
INFERENCE_API_KEY=your_key
```

Only the keys for providers you use are needed. Web retrieval and hosted model calls require network access.
Endpoints such as OpenAI, Openouter, and Interweb are also implemented and available for processing knowledge and answering questions. `inference` and `interweb` uses the configured university endpoint.

## Cutural knowledge and question generation

The pipeline retrieves documents in English and the culture's local language, extracts knowledge in batches, and generates factual, conceptual, misleading, and long-form questions for the set of cultures in main.py.
To add a new culture, simply include it in the `cultures` dictionary, along with its corresponding language in `culture_language`. Both can be found in the main.py.
Each dimension completes its processing and saves its results before the next one begins.

To start a run, you must select the mode to use. By default, `inference` is set as the endpoint provider.

| Mode | Behavior |
| --- | --- |
| `full` | Starts from the beginning, clears previous data in the `results` folder and creates the complete dimension queue in dimensions_queue.json. |
| `resume` | Continues `dimensions_queue.json`. Each queued dimension starts again. |
| `retry` | Reads `emptyKnowledge.json` and generates questions for failed dimensions. |
| `single` | Runs selected cultures and dimensions configured in the `single` branch of `src/main.py`. Clears previous results like `full`. |

```powershell
# Start a new complete run considering all 140 dimensions
python -m src.main --mode full  

# Continue after an interruption
python -m src.main --mode resume  

# Retry recorded dimensions failures
python -m src.main --mode retry  

# Test selected dimensions after editing the single-mode lists
python -m src.main --mode single 
```

Use the same API, language, and question-type settings when resuming or retrying.
An item leaves the queue only after its results or failure report are saved.
Interrupting a dimension leaves it queued; resume processes that entire dimension
again. Failed attempts go to `emptyKnowledge.json` for retry.

### Generation arguments

| Argument | Values / default | Purpose |
| --- | --- | --- |
| `--mode` | `full` (default), `resume`, `retry`, `single` | Select the workflow. |
| `--api` | `inference` (default), `openai`, `openrouter`, `interweb` | Select the model provider. |
| `--batch_size` | Positive integer; default `2` | Documents per knowledge batch; all valid documents are processed. |
| `--question_language` | `english` (default), `local` | Select the question language. |
| `--question_type` | `all` (default), `factual`, `conceptual`, `misleading`, `long_form` | Generate all four types or one type. |
Teacher models are configured in `TEACHER_MODELS` in `src/main.py`.

## Model examination

Finish question generation before starting examination.

Examination reads `results/questions/questions.json`, asks the configured student models to answer, and scores their responses against reference answers. Configure `STUDENT_MODELS`, `GRADER_MODELS`, and `GRADER_ENDPOINT` in `examinationLLM.py`.


```powershell
# Start a fresh examination; clears previous examination results
python -m examinationLLM --mode full

# Complete failed or unvisited questions after an interrupted examination
python -m examinationLLM --mode retry

# Update with new results from saved answers without new model calls
python -m examinationLLM --mode rebuild_metrics
```

## Graphics

After examination is complete:

```powershell
python -m graphics
```

This generates graphics and saves accuracy charts by culture, question type, and cultural category, plus a heatmap, in `results/exam/`. Graph generation requires
complete examination results and equal evaluated question counts across models to assure consitency.

## Batch-size experiment

Compare knowledge generation with different batch sizes using saved retrieval
results, without generating questions:

```powershell
python -m backup_scripts.batch_experiment --api openrouter --batch_sizes 2 3 5
```

Optional arguments are `--input` (default
`results/knowledge/query_results.json`), `--api` (`interweb` by default; also
`openai` or `openrouter`), and `--llm_model` (`gpt-4o` by default).
Each experiment is saved separately in `experiment/<timestamp>/`.

## Results and project files

| Location | Contents |
| --- | --- |
| `results/knowledge/` | Retrieval snapshots, knowledge input/output, raw knowledge responses, `dimensions_queue.json`, and `emptyKnowledge.json`. |
| `results/questions/` | Accepted questions in `questions.json`, raw responses in `questions_raw.json`, and `<Culture>_Knowledge_QA.csv` exports. |
| `results/exam/` | Answers and scores in `exam_results.json`, run configuration, `failed_questions.json`, metric CSVs, terminal logs, and PNG charts. |
| `results/` | Timestamped generation summaries and terminal logs. |
| `cultural_parameters/` | Dimension definitions and cultural categories. |
| `src/main.py` | Generation entry point and culture/model configuration. |
| `web_retrieval.py`, `knowledgeGen.py`, `questionGen.py` | Retrieval, knowledge extraction, and question generation. |
| `examinationLLM.py`, `studentLLM.py`, `graphics.py` | Evaluation, student/grader calls, and charts. |
| `utils.py`, `result_paths.py` | Shared helpers, queue persistence, and output paths. |