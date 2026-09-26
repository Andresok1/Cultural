from utils import update_empty_knowledge_report
from result_paths import KNOWLEDGE_DIR, QUESTIONS_DIR
import re
import random
from time import perf_counter
import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
import json
import os
from dotenv import load_dotenv
import pandas as pd
import requests

QUESTION_TYPES = {
    "factual": "Based on the context, think through all relevant cultural points step by step and generate a factual question. The question type can include single-choice, true/false, or fill-in-the-blank. Ensure that the question stem is clear, the options are plausible but misleading (distractors), and the answer is accurate.",
    "conceptual": "Based on the context, think through all relevant cultural points step by step and generate a conceptual explanation question. The question should focus on the learner’s understanding of the concepts, structures, or values inside cultural phenomena, rather than simple memorization. Suitable formats include multiple-choice or true/false questions. Ensure the question is thought-provoking and the answer is well-justified.",
    "misleading": "Based on the context, think through all relevant cultural points step by step and generate a misleading question to assess whether learners can identify cultural misunderstandings, stereotypes, or biases. The question should focus on learners’ critical thinking about culture, identifying which statements or Behaviors reflect misunderstandings, oversimplifications, biases, or stereotypes, and guide them toward more accurate or respectful understandings. Possible formats include multiple-choice, true/false, case analysis, or short-answer questions.",
    "multihop": "Based on the context, think through all relevant cultural points step by step and generate a multi-hop reasoning question to assess whether the learner can synthesize multiple cultural elements and understand the deeper logic or internal connections among cultural phenomena. The question should prompt learners to start from multiple information points, integrate cultural knowledge, and perform logical analysis, comparison, or generalization. Scenario-based, integrated analysis, or comparative reasoning questions are recommended."
}

ROLE_QUESTION = """
You are an expert educational assessment designer specialized in cultural knowledge evaluation.
Your task is to create accurate assessment questions from provided cultural information.
You design items that evaluate understanding, reasoning, and interpretation.
Questions may include multiple-choice, true/false, short-answer, or long-answer formats.
"""

def group_raw_questions(history):
    """Migrate older raw files to culture -> dimension -> type -> responses."""
    if isinstance(history, dict):
        for dimensions in history.values():
            for dimension, responses in dimensions.items():
                if isinstance(responses, list):
                    dimensions[dimension] = {"Unknown": responses}
        return history
    if not isinstance(history, list):
        history = [{"response": history}]
    grouped = {}
    for entry in history:
        if not isinstance(entry, dict):
            entry = {"response": entry}
        culture = entry.get("culture") or "Unknown"
        dimension = entry.get("dimension") or "Unknown"
        question_type = entry.get("question_type") or "Unknown"
        grouped.setdefault(culture, {}).setdefault(dimension, {}).setdefault(question_type, []).append(entry["response"])
    return grouped


def save_raw_question(response, *, culture, question_type, dimension=None):
    """Keep every original response, including empty answers and retry attempts."""
    output_path = QUESTIONS_DIR / "questions_raw.json"
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            history = json.load(f)
        history = group_raw_questions(history)
    else:
        history = {}

    history.setdefault(culture, {}).setdefault(dimension or "Unknown", {}).setdefault(question_type, []).append(response)
    temporary_path = output_path.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def random_llm(selected_format):

    randomness_prompt = ""
    reference = ""
    if selected_format == "single_choice":
        reference= random.choice(["A","B","C","D"])
        randomness_prompt = f"""
            The correct answer must be the option: {reference}
        """
    elif selected_format== "true_false":
        reference= random.choice(["True","False"])
        randomness_prompt = f"""
            The correct answer must be the option: {reference}
            For True and False questions tailor your question to match this.
        """
    return randomness_prompt
     
    

def PROMPT_QUESTION(instruction, prompt_texts, question_type, language="english", ):

    question_formats = {

        "single_choice": """
        Question: [question]
        Options:
        A) [option_a]
        B) [option_b]
        C) [option_c]
        D) [option_d]

        Reference Answer: [answer]
        """,

        "true_false": """
        Question: Is the following statement true or false? [question]
        Options: "NA"
        Reference Answer: [answer]
        """,

        "fill_the_blank": """
        Question: Complete the sentence: [question]
        Options: "NA"
        Reference Answer: [answer]
        """,

        "short_answer": """
        Question: [question]
        Options: "NA"
        Reference Answer: [answer]

        """,

        "long_answer": """
        Question: [question]
        Options: "NA"
        Reference Answer: [answer]

        """
    }

    question_types = {
        "factual": 
            [
                "single_choice",
                "true_false",
                "fill_the_blank"
            ],

        "conceptual": 
            [
                "single_choice",
                "true_false",
            ],

        "misleading": 
            [
                "single_choice",
                "true_false",
                "short_answer"
            ],

        "multihop": 
            [
                "long_answer"
            ],
    }   

    permited_question_types = question_types[question_type]

    random.seed()

    selected_format = random.choice(permited_question_types)

    question_format = question_formats[selected_format]


    prompt = f"""
        Task: Answer in {language}.
        Instruction:
        {instruction}

        Questions have to be clear.
        The options should be plausible but misleading distractors.
        The answer should be accurate.
        Reference Answer must indicate the correct option.

        Context:
        {prompt_texts}
        
        Question type is: {selected_format}
        The question should be in the following format:
        {question_format}

        IMPORTANT:
        1. The question should avoid explicitly mentioning cultural concepts, terminology, or characteristics, in order to effectively assess the student’s understanding of cultural traits.
        2. A reference answer should be provided after the question.
        3. Do not change the structure of format given. Just fill in the content after these labels.
        4. If the provided information is insufficient to generate a meaningful question, do not ask for clarification and do not provide an explanation. Instead, keep the exact same format and write "EMPTY" in all fields.
        5. STRICT OUTPUT FORMAT: Replace only the content inside the brackets []. Your response must contain only the final values that belong inside those brackets. Do not provide reasoning, explanations, justifications, summaries, introductions, conclusions, markdown, or any additional text. Any text outside the required placeholders will be considered an invalid response.

        Now generate the output.
        """

    if selected_format == "single_choice" or  selected_format == "true_false":
        random_feature= random_llm(selected_format)
        prompt = prompt + random_feature

    if selected_format in ["short_answer", "long_answer"]:
        prompt += """
        IMPORTANT FOR OPEN-ENDED QUESTIONS:
        The Reference Answer must contain a complete substantive answer.
        Never write "NA", or leave the Reference Answer blank.
        Only the Options field should be "NA".
        """
    
    return prompt, selected_format

def openai_create_question(text, question_type, culture, question_language, dimension=None):
    """
    As input it receives a text and extracts important features and content related to a specific culture to generate a question.
    
    Returns format: title, snippet and knowledge extracted from the text.
    Does not invent information if there is insufficient support.
    """
    openai_model = "gpt-4o-mini"

    if not text:

        return  None
    
    
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Context {i}:\n{text}" for i, text in enumerate(texts, 1)])

    
    if question_language == "local":
         
        local_dictionary = {
            "colombian": "Spanish",
            "italian": "Italian",
            "german": "German"}

        language = local_dictionary.get(culture, "English")
    else:        
        language = "English"

    instruction = QUESTION_TYPES[question_type]
    prompt, selected_format= PROMPT_QUESTION(instruction, prompt_texts, question_type, language)

    load_dotenv()

    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )

    response = client.chat.completions.create(
        model= openai_model, 
        response_format={"type":"json_object"},
        messages=[{"role": "user", "content": ROLE_QUESTION + prompt}]
    )

    content = response.choices[0].message.content
    save_raw_question(content, culture=culture, dimension=dimension, question_type=question_type)
    return content, selected_format

def interweb_create_question(args, text, question_type, culture, question_language, retries=5, dimension=None):
    """ 

    """
    interweb_model = "gpt-4o-mini"
    if not text:
        print("No documents were given to produce a question.")
        return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Context {i}:\n{text}" for i, text in enumerate(texts, 1)])

    
    if question_language == "local":
         
        local_dictionary = {
            "colombian": "Spanish",
            "italian": "Italian",
            "german": "German"}

        language = local_dictionary.get(culture, "English")
    else:        
        language = "English"


    instruction = QUESTION_TYPES[question_type]

    prompt, selected_format = PROMPT_QUESTION(instruction, prompt_texts, question_type, language)

    load_dotenv()
    INTERWEB_API_KEY = os.getenv("INTERWEB_API_KEY")
    url = "https://interweb.l3s.uni-hannover.de"
    
    headers = {
        "Authorization": f"Bearer {INTERWEB_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }

    payload = {
        "model": interweb_model, #Model can be changed.
        "messages": [
            {
                "role": "system",
                "content": ROLE_QUESTION,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }

    try:
            
        response = requests.post(
            f"{url}/v1/chat/completions",
            response_format={"type":"json_object"},
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            print("Error:", response.status_code, response.text)

            if retries > 0:
                print(f"Retrying... attempts left: {retries}")
                return interweb_create_question(
                    args, 
                    text, 
                    question_type, 
                    culture, 
                    question_language, 
                    retries - 1, dimension=dimension
                )   

            return None
        
        answer = response.json()["choices"][0]["message"]["content"]
        save_raw_question(answer, culture=culture, dimension=dimension, question_type=question_type)

        if answer is None or "[]" in answer or answer.strip() == "":
            if retries > 0:
                print(f"Empty response. Retrying... attempts left: {retries}")
                return interweb_create_question(
                    args, 
                    text, 
                    question_type, 
                    culture, 
                    question_language, 
                    retries - 1, dimension=dimension
                )

            return None

        return answer, selected_format
    
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

        if retries > 0:
            return interweb_create_question(
                args, 
                text, 
                question_type, 
                culture, 
                question_language, 
                retries - 1, dimension=dimension
            )

        return None

def openrouter_create_question(args, text, question_type, culture, dimension, question_language, retries=5):

    openrouter_model = "openai/gpt-4o"
     
    if not text:
            print(f"No documents were given to produce a knowledge entry-> {culture}, {dimension}.")

            return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Context {i}:\n{text}" for i, text in enumerate(texts, 1)])

    if question_language == "local":
         
        local_dictionary = {
            "colombian": "Spanish",
            "italian": "Italian",
            "german": "German"}

        language = local_dictionary.get(culture, "English")
    else:        
        language = "English"


    instruction = QUESTION_TYPES[question_type]

    prompt, selected_format = PROMPT_QUESTION(instruction, prompt_texts, question_type, language)

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    data = {
        "model": openrouter_model,
        "messages": [
            {
                "role": "system",
                "content": ROLE_QUESTION,
            },
            {
                "role": "user",
                "content": f"{prompt}\n"
            }
        ]
    }

    response = requests.post(
        f"{url}/api/v1/chat/completions",
        response_format={"type":"json_object"},
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]
    save_raw_question(answer, culture=culture, dimension=dimension, question_type=question_type)

    return answer, selected_format



def inference_create_question(args, text, question_type, culture, dimension, question_language, retries=2, language=None):
    inference_model = "granite-4.1:8b-bf16"

    load_dotenv()
    INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY")

    if not text:
        print(f"No documents were given to produce a knowledge entry-> {culture}, {dimension}.")
        return None

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

    instruction = QUESTION_TYPES[question_type]

    prompt, selected_format= PROMPT_QUESTION(instruction, prompt_texts, question_type, language)

    client = OpenAI(
        base_url=f"https://inference.kbs.uni-hannover.de/v1",
        api_key=os.getenv("INFERENCE_API_KEY"),
        timeout=300,
    )

    for attempt in range(retries + 1):
        try:

            print(f"Question |{language + ' | ' if language else ''}inference / {inference_model} | Sending request...", flush=True)

            started = perf_counter()

            response = client.chat.completions.create(
                model=inference_model,
                response_format={"type":"json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": ROLE_QUESTION
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n"
                    }
                ]
            )

            answer = response.choices[0].message.content

            print(f"Question |{language + ' | ' if language else ''}inference / {inference_model} | Finished after {perf_counter() - started:.1f}s")

            if (
                answer is None
                or "[]" in answer
                or answer.strip() == ""
            ):

                print("Empty response from inference.")

                if attempt < retries:
                    wait_time = min(2 ** attempt, 60)

                    print(f"Retrying in {wait_time}s...")
                        

                    time.sleep(wait_time)
                    continue

                print(
                    f"Game Over: empty response for "
                    f"{culture} | {dimension} | {language}"
                )

                return None

            save_raw_question(answer, culture=culture, dimension=dimension, question_type=question_type)

            return answer, selected_format

        except (
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError
        ) as e:

            print(
                f"API error on attempt "
                f"{attempt + 1}/{retries + 1}:"
            )

            print(
                f"Timeout after {perf_counter()-started:.1f}s"
            )

            print(f"{type(e).__name__}: {e}")

            if attempt >= retries:

                print(f"Game Over: {culture} | {dimension} | {language}")

                return None

            wait_time = min(15 * (2 ** attempt), 60)

            print(f"Retrying in {wait_time}s...")                

            time.sleep(wait_time)

        except Exception as e:

            print(
                f"Unexpected error: "
                f"{type(e).__name__}: {e}"
            )

            return None

    return None


def knowledge_preparing(args, culture, dimension, knowledge_output_dict):
    """Prepare valid knowledge and record dimensions empty across all languages."""

    knowledge_list = []
    title_list = []
    snippet_list = []

    for lang, knowledge_set in knowledge_output_dict[culture][dimension].items():
        try:
            items = json.loads(knowledge_set) if isinstance(knowledge_set, str) and knowledge_set.strip() else (knowledge_set or [])
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON for {culture} | {dimension} | {lang}, skipping")
            continue

        if isinstance(items, dict):
            items = [items]

        if not isinstance(items, list):
            print(f"Warning: invalid knowledge format for {culture} | {dimension} | {lang}, skipping")
            continue

        for data in items:
            if not isinstance(data, dict):
                continue

            know = data.get("Knowledge") or data.get("knowledge")
            titl = data.get("Title") or data.get("title")
            snipp = data.get("Snippet") or data.get("snippet")

            if isinstance(know, list):
                know = ", ".join(map(str, know))

            if not all(isinstance(v, str) and v.strip() and v.upper() != "EMPTY" for v in (know, titl, snipp)):
                continue

            if "NOT RELEVANT INFORMATION" in titl.upper():
                continue

            knowledge_list.append(know.replace(";", ",").strip())
            title_list.append(titl.replace(";", ",").strip())
            snippet_list.append(snipp.replace(";", ",").strip())

    print(f"Question | {culture} | {dimension} | Knowledge entries: {len(knowledge_list)}")


    return knowledge_list, title_list, snippet_list

def knowledge_to_question(args, culture, dimension, knowledge_list, typ):
    '''This function separates title, snippet and knowledge from the knowledge_path and clean them. and separates them into lists to better visualization and data control to create a question.
    It gives knowledge_list, title_list and snippet_list scaning the knowledge_path.
    It creates at the end the question_reference_{timestamp}.json file with the question, options and reference answer.

    return:
    question_cleaned, abcd_options_cleaned, reference_answer, knowledge_list, title_list, snippet_list, selected_format
    '''
    output_path = QUESTIONS_DIR / "questions.json"
            
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            question_vector = json.load(f)
    else:
        question_vector = {}

    if culture not in question_vector:
        question_vector[culture] = {}

    if dimension not in question_vector[culture]:
        question_vector[culture][dimension] = {}

    started = perf_counter()
    if args.api == "openai":
        result = openai_create_question(text=knowledge_list, question_type=typ, culture=culture, question_language=args.question_language, dimension=dimension)
    elif args.api == "openrouter":
        result = openrouter_create_question(args, text=knowledge_list, question_type=typ, culture=culture,dimension=dimension, question_language=args.question_language)
    elif args.api == "inference":
        result = inference_create_question(args, text=knowledge_list, question_type=typ, culture=culture, dimension=dimension, question_language=args.question_language)
    else:
        result = interweb_create_question(args, text=knowledge_list, question_type=typ, culture=culture, question_language=args.question_language, dimension=dimension)

    if result is None:
        print(f"{typ} | No question generated.\n")
        selected_format = None
        question_reference = None
    else: 
        question_reference, selected_format = result

    #for each question type there is a differnet format to follow
    if question_reference is not None:
        try:    
            question_data = json.loads(question_reference)

        except json.JSONDecodeError:
            print("Invalid JSON response:")
            print(question_reference)

            question_cleaned = "EMPTY"
            abcd_options_cleaned = "EMPTY"
            reference_answer = "EMPTY"

        else:

            question_cleaned = question_data.get("Question", "EMPTY").strip()

            if selected_format == "short_answer":
                question_cleaned += " Write a concise answer to the question above. Expected answer length: 3-5 sentences."
            elif selected_format == "long_answer":
                question_cleaned += " Write a detailed essay answering the question above. Explain your answer clearly and provide the main reasons supporting it. Expected answer length: 5-8 sentences."

            reference_answer = question_data.get("Reference Answer", "EMPTY").strip()

            if selected_format == "single_choice":

                options = question_data.get("Options", {})

                abcd_options_cleaned = " ".join(
                    [
                        f"{key}) {value}"
                        for key, value in options.items()
                    ]
                )

            else:

                abcd_options_cleaned = "NA"


    else:
        question_cleaned = "EMPTY"
        abcd_options_cleaned = "EMPTY"
        reference_answer = "EMPTY"


    if typ not in question_vector[culture][dimension]:
        question_vector[culture][dimension][typ] = []

    if selected_format == "single_choice":
        question_vector[culture][dimension][typ].append({
            "format": selected_format,
            "question": question_cleaned,
            "options": abcd_options_cleaned,
            "reference_answer": reference_answer    
        })
    else:
        question_vector[culture][dimension][typ].append( {
            "format": selected_format,
            "question": question_cleaned,
            "options": abcd_options_cleaned,
            "reference_answer": reference_answer
        })


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question_vector, f, ensure_ascii=False, indent=2)

    if question_reference is not None:
        print(f"{typ:<10} | Finished after {perf_counter() - started:.1f}s", flush=True)

    return question_cleaned, abcd_options_cleaned, reference_answer, selected_format



def csv_saver(args, dimension, culture, timestamp, culture_dfs, knowledge_output_dict):
    '''This function prepares the question data to deliver it in `.csv` format and save it in the `results` folder.
    return:
        Counts of requested and successfully generated questions.
    '''

    knowledge_list, title_list, snippet_list= knowledge_preparing(args, culture, dimension, knowledge_output_dict)
    question_counts = 0

    if not knowledge_list:
        print("Skipped: no relevant knowledge.")
        return question_counts

    if args.question_type == "all":
        types = ["factual", "conceptual", "misleading", "multihop"]
    else: 
        types = [args.question_type]

    for typ in types:
        question, abcd_options, reference_answer, selected_format = knowledge_to_question(args, culture, dimension,knowledge_list=knowledge_list, typ=typ)

        checks = {
            "question_not_string": not isinstance(question, str),
            "question_empty": not question.strip(),
            "question_is_EMPTY": question.strip().upper() == "EMPTY",
            "question_not_enough_information": "NOT ENOUGH INFORMATION" in question.upper(),
            "answer_not_string": not isinstance(reference_answer, str),
            "answer_empty": not reference_answer.strip(),
            "answer_NA": (reference_answer.strip().upper() == "NA"),
            "answer_is_EMPTY": reference_answer.strip().upper() == "EMPTY",
            "answer_not_enough_information": "NOT ENOUGH INFORMATION" in reference_answer.upper()
        }

        failed = [name for name, result in checks.items() if result]

        if failed:
            print("Question rejected because:", failed)

            update_empty_knowledge_report(culture, dimension)

        else:
            question_counts += 1

        output_path = KNOWLEDGE_DIR / f"knowledge_output_{timestamp}.json"

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                knowledge_vector = json.load(f)
        else:
            knowledge_vector = {}

        df_dimension = pd.DataFrame([{"culture": culture, "dimension": dimension}])

        data_knowledge_info = {} 
        for i, (t, s, k) in enumerate(zip(title_list, snippet_list, knowledge_list), start=1):
            data_knowledge_info[f"title_{i}"] = [t]
            data_knowledge_info[f"snippet_{i}"] = [s]
            data_knowledge_info[f"knowledge_{i}"] = [k]

        df_knowledge_info = pd.DataFrame(data_knowledge_info)

        if culture not in knowledge_vector:
            knowledge_vector[culture] = {}


        knowledge_vector[culture][dimension] = data_knowledge_info

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_vector, f, ensure_ascii=False, indent=2)

        ### QUESTION PREPARATION 
        if selected_format == "single_choice":
            df_questions_reference = pd.DataFrame({
                "question": [question],
                "abcd_options": [abcd_options],
                "reference_answer": [reference_answer]
            })
        else:  
            df_questions_reference = pd.DataFrame({
                "question": [question],
                "reference_answer": [reference_answer]
            })


        ### .CSV preparation and saving 
        df = pd.concat([df_dimension.reset_index(drop=True),
                        df_knowledge_info.reset_index(drop=True),
                        df_questions_reference.reset_index(drop=True)], axis=1)

        culture_dfs[culture].append(df)


        for culture, dfs in culture_dfs.items():
            if dfs:  
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.to_csv(QUESTIONS_DIR / f"{culture}_Knowledge_QA.csv", index=False, encoding="utf-8-sig")
            else:
                print(f"Warning: No data to save for culture {culture}")

    return question_counts


def parse_question_reference(question_reference, selected_format):
    """
    Parse question reference text into question, options, and reference answer.

    Parameters:
        question_reference (str): Raw question reference text.
        selected_format (str): Question type format.

    Returns:
        tuple: question_cleaned, abcd_options_cleaned, reference_answer
    """

    question_reference = question_reference.split("Question:", 1)[1]

    # Special handling for single choice questions
    if selected_format == "single_choice":

        parts = question_reference.split("Reference Answer:", 1)

        question_text = (
            parts[0]
            .replace("Question:", "")
            .replace("Options:", "")
            .strip()
        )

        split_index = question_text.find("A)")
        if split_index == -1:
            split_index = question_text.find("a)")

        question = question_text[:split_index].strip()

        question_cleaned = question.replace("\n", " ")

        abcd_options = question_text[split_index:].strip()

        abcd_options_cleaned = (
            abcd_options
            .replace("\n", " ")
            .replace("  ", " ")
        )

        reference_answer = parts[1].strip()

    else:

        question_part, reference_answer = question_reference.split(
            "Reference Answer:", 1
        )

        question_text = question_part.split("Options:", 1)[0]

        question_cleaned = " ".join(question_text.split())

        abcd_options_cleaned = "NA"

        reference_answer = " ".join(reference_answer.split())

    return question_cleaned, abcd_options_cleaned, reference_answer
