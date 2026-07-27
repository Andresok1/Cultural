import re

from openai import OpenAI
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

ROLE_QUESTION = "You are an expert educational assessment designer specialized in cultural knowledge evaluation. Your task is to create accurate multiple-choice questions from provided cultural information. You design assessment items that evaluate understanding, reasoning, and interpretation of cultural traits. You must ensure questions are clear, unbiased, and supported by the provided context."

def PROMPT_QUESTION(idiom, instruction, prompt_texts):
    
    prompt = f"""
        Task: Answer in {idiom}.
        Instruction:
        {instruction}
        The correct answer must be randomly placed among A, B, C, and D.
        The position of the correct answer should vary across generated questions.

        Questions have to be clear.
        The options should be plausible but misleading distractors.
        The answer should be accurate.
        Reference Answer must indicate the correct option.

        Note:
        1. The question should avoid explicitly mentioning cultural concepts, terminology,
        or characteristics, in order to effectively assess the student’s understanding
        of cultural traits.
        2. A reference answer should be provided after the question.
        3. Do not change the structure of "Question", "Options" and "Reference Answer" in the output, as they will be used for evaluation. Just fill in the content after these labels.
        Context:
        {prompt_texts}
        
        The question should be in the following format:
        Question: ...
        A) ...
        B) ...
        C) ...
        D) ...

        Reference Answer: X
        """
    return prompt


def openai_create_question(text, question_type, culture, question_language):
    """
    As input it receives a text and extracts important features and content related to a specific culture to generate a question.
    
    Returns format: title, snippet and knowledge extracted from the text.
    Does not invent information if there is insufficient support.
    """

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

        idiom = local_dictionary.get(culture.lower(), "English")
    else:        
        idiom = "English"

    instruction = QUESTION_TYPES[question_type]

    prompt= PROMPT_QUESTION(idiom, instruction, prompt_texts)

    load_dotenv()

    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4.1-mini",   #OPENAI constant Model
        messages=[{"role": "user", "content": ROLE_QUESTION + prompt}]
    )

    content = response.choices[0].message.content
    return content

def interweb_create_question(args, text, question_type, culture, question_language):
    """ 

    """
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

        idiom = local_dictionary.get(culture.lower(), "English")
    else:        
        idiom = "English"

    instruction = QUESTION_TYPES[question_type]

    prompt= PROMPT_QUESTION(idiom, instruction, prompt_texts)

    load_dotenv()
    INTERWEB_API_KEY = os.getenv("INTERWEB_API_KEY")
    url = "https://interweb.l3s.uni-hannover.de"
    
    headers = {
        "Authorization": f"Bearer {INTERWEB_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }

    payload = {
        "model": args.llm_model, #Model can be changed.
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
                    retries - 1
                )   

            return None
        
        answer = response.json()["choices"][0]["message"]["content"]

        if answer is None or "[]" in answer or answer.strip() == "":
            if retries > 0:
                print(f"Empty response. Retrying... attempts left: {retries}")
                return interweb_create_question(
                    args, 
                    text, 
                    question_type, 
                    culture, 
                    question_language, 
                    retries - 1
                )

            return None

        return answer
    
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

        if retries > 0:
            return interweb_create_question(
                args, 
                text, 
                question_type, 
                culture, 
                question_language, 
                retries - 1
            )

        return None


def knowledge_to_question(args, culture, dimension, knowledge_output_dict):
    '''This function separates title, snippet and knowledge from the knowledge_path and clean them. and separates them into lists to better visualization and data control to create a question.
    It gives knowledge_list, title_list and snippet_list scaning the knowledge_path.
    It creates at the end the question_reference_{timestamp}.json file with the question, options and reference answer.

    return:
    question_cleaned, abcd_options_cleaned, reference_answer, knowledge_list, title_list, snippet_list
    '''
    output_path = f"results/question_reference_{timestamp}.json"
    

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            question_vector = json.load(f)
    else:
        question_vector = {}

    if culture not in question_vector:
        question_vector[culture] = {}

    knowledge_list = []
    title_list = []
    snippet_list = []
    notEnoughInfo = []


    knowledge_items = knowledge_output_dict[culture][dimension]


    print(f"Procesing question to '{dimension}' in '{culture}'") 

    
    for index, knowledge_set in enumerate(knowledge_items):

        if not knowledge_set or not knowledge_set.strip():
            print("Warning: empty knowledge_set in some language Query (english /local), skipping")
            missing_know = (f"Missing Knowledge'{dimension}' in '{culture}' culture:")
            if index == 0:
                print(missing_know + "EN")
            else:
                print(missing_know + "Local Language")
            continue

        try:
            item = json.loads(knowledge_set)  # it converts the string back to a dictionary 
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON, skipping: {knowledge_set[:50]}...")
            continue

        for data in item:
            if data: 
                know = data.get('Knowledge') or data.get('knowledge')
                titl = data.get('Title') or data.get('title')
                snipp = data.get('Snippet') or data.get('snippet')
            else: 
                know = None
                titl = None
                snipp = None

            if know:
                know_cleaned = ", ".join(know) if isinstance(know, list) else str(know)
                know_cleaned = know_cleaned.replace(";", ",").strip()
                knowledge_list.append(know_cleaned)
            else: 
                knowledge_list.append("EMPTY")


            if titl:
                title_cleaned = titl.replace(";", ",").strip()
                title_list.append(title_cleaned)
            else:
                title_cleaned = ""
                title_list.append("EMPTY")


            if snipp:
                snippet_cleaned = snipp.replace(";", ",").strip()
                snippet_list.append(snippet_cleaned)
            else:
                snippet_list.append("EMPTY")

            if not titl or "(info missing)" in title_cleaned:
                print("Added to notEnoughInfo_dimension: ",title_cleaned)
                notEnoughInfo.append(f"{dimension} in {culture}")

    print(f"El Knowledge es de: {len(knowledge_list)} unidades")
    print("\n")


    if notEnoughInfo is not None:
        print("notEnoughInfo_dimension:", notEnoughInfo)  #this should be empty if all is working

    if args.api == "openai":
        question_reference = openai_create_question(text=knowledge_list, question_type="factual", culture=culture, question_language=args.question_language)
    else: 
        question_reference= interweb_create_question(args, text=knowledge_list, question_type="factual", culture=culture, question_language=args.question_language)

    
    if question_reference is not None:
        question_reference = question_reference.split("Question:", 1)[1]

        parts = question_reference.split("Reference Answer:", 1)

        question_text = parts[0].replace("Question:", "").strip()
        
        split_index = question_text.find("A)")
        if split_index == -1:
            split_index = question_text.find("a)")

        question = question_text[:split_index].strip()
        question_cleaned = question.replace("\n", " ")

        abcd_options = question_text[split_index:].strip()
        abcd_options_cleaned = abcd_options.replace("\n", " ").replace("  ", " ")

        reference_answer = parts[1].strip()

    else:
        question_cleaned = "EMPTY"
        abcd_options_cleaned = "EMPTY"
        reference_answer = "EMPTY"


    question_vector[culture][dimension] = {
        "question": question_cleaned,
        "options": abcd_options_cleaned,
        "reference_answer": reference_answer
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question_vector, f, ensure_ascii=False, indent=2)

    return question_cleaned, abcd_options_cleaned, reference_answer, knowledge_list, title_list, snippet_list



def csv_saver(args, dimension, culture, timestamp, culture_dfs, knowledge_path, knowledge_output_dict):
    '''This function prepares the question data to deliver it in `.csv` format and save it in the `results` folder.
    return:
        None.
    '''


    question, abcd_options, reference_answer, knowledge_list, title_list, snippet_list, = knowledge_to_question(args, culture, dimension, knowledge_output_dict)

    output_path = f"results/knowledge_output_{timestamp}.json"

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            knowledge_vector = json.load(f)
    else:
        knowledge_vector = {}

    df_dimension = pd.DataFrame([{"culture": culture, "dimension": dimension}])



    data_knowledge_info = {}    #TODO: The structure can be in groups of 3. 
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
    df_questions_reference = pd.DataFrame({
        "question": [question],
        "abcd_options": [abcd_options],
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
            final_df.to_csv(f"results/{culture}_Knowledge_QA.csv", index=False, encoding="utf-8-sig")
        else:
            print(f"Warning: No data to save for culture {culture}")
