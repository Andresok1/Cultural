import re
import random

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

def random_llm(selected_format):

    randomness_prompt = ""
    reference = "EMPTY"
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
    print("Reference answer:", reference)
    print("\n")
    return randomness_prompt
     
    


def PROMPT_QUESTION(language, instruction, prompt_texts, question_type):

    question_formats = {

        "single_choice": """
        Question: {question}
        Options:
        A) {option_a}
        B) {option_b}
        C) {option_c}
        D) {option_d}

        Reference Answer: {answer}
        """,

        "true_false": """
        Question: Is the following statement true or false? {question}
        Options: "NA"
        Reference Answer: {answer}
        """,

        "fill_the_blank": """
        Question: Complete the sentence: {question}
        Options: "NA"
        Reference Answer: {answer}
        """,

        "short_answer": """
        Question: Write a short essay answering the following question. Expected answer length: 3-5 sentences. {question}. 
        Options: "NA"
        Reference Answer: {answer}

        """,

        "long_answer": """
        Question: Write an essay answering the following question and also explain the reasoning step by step. Expected answer length: 5-8 sentences. {question}
        Options: "NA"
        Reference Answer:
        {answer}

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

        Note:
        1. The question should avoid explicitly mentioning cultural concepts, terminology, or characteristics, in order to effectively assess the student’s understanding of cultural traits.
        2. A reference answer should be provided after the question.
        3. Do not change the structure of format given. Just fill in the content after these labels.
        Context:
        {prompt_texts}
        
        Question type is: {selected_format}
        The question should be in the following format:
        {question_format}

        """
    print("question type:", question_type)
    print("format:", selected_format)

    if selected_format == "single_choice" or  selected_format == "true_false":
        random_feature= random_llm(selected_format)
        prompt = prompt + random_feature
    
    return prompt, selected_format

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

        language = local_dictionary.get(culture, "English")
    else:        
        language = "English"

    if question_type == "random":
        question_type = random.choice(list(QUESTION_TYPES.keys()))

    instruction = QUESTION_TYPES[question_type]

    prompt, selected_format= PROMPT_QUESTION(language, instruction, prompt_texts, question_type)

    load_dotenv()

    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4.1-mini",   #OPENAI constant Model
        messages=[{"role": "user", "content": ROLE_QUESTION + prompt}]
    )

    content = response.choices[0].message.content
    return content, selected_format

def interweb_create_question(args, text, question_type, culture, question_language, retries=5):
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

        language = local_dictionary.get(culture, "English")
    else:        
        language = "English"

    if question_type == "random":
        question_type = random.choice(list(QUESTION_TYPES.keys()))

    instruction = QUESTION_TYPES[question_type]

    prompt, selected_format = PROMPT_QUESTION(language, instruction, prompt_texts, question_type)

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
                retries - 1
            )

        return None

def openrouter_create_knowledge(args, text, question_type, culture, dimension, question_language, retries=5):

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

    if question_type == "random":
        question_type = random.choice(list(QUESTION_TYPES.keys()))

    instruction = QUESTION_TYPES[question_type]

    prompt, selected_format = PROMPT_QUESTION(language, instruction, prompt_texts, question_type)

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    data = {
        "model": f"openai/{args.llm_model}",
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
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]

    return answer, selected_format


# def openrouter_create_question(
#     args,
#     text,
#     question_type,
#     culture,
#     question_language,
#     retries=5
# ):
#     """
#     Generates questions using OpenRouter API.
#     """

#     load_dotenv()

#     OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

#     url = "https://openrouter.ai/api/v1/chat/completions"

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": "https://your-project-url.com",
#         "X-Title": "Question Generation Pipeline"
#     }


#     user_prompt = PROMPT_QUESTION(
#         question_type,
#         culture,
#         question_language,
#         text
#     )


#     print("     OpenRouter API is using:", args.llm_model)


#     payload = {
#         "model": args.llm_model,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": ROLE_QUESTION
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt
#             }
#         ],
#         "temperature": 0.3,
#         "max_tokens": 2048
#     }


#     try:

#         response = requests.post(
#             url,
#             headers=headers,
#             json=payload,
#             timeout=120
#         )


#         # Debug útil
#         print("Status:", response.status_code)

#         if response.status_code != 200:
#             print("OpenRouter error:")
#             print(response.text)

#             if retries > 0:
#                 print(
#                     f"Retrying... attempts left: {retries}"
#                 )

#                 return openrouter_create_question(
#                     args,
#                     text,
#                     question_type,
#                     culture,
#                     question_language,
#                     retries - 1
#                 )

#             return None


#         data = response.json()


#         # Comprobar que existe respuesta
#         if (
#             "choices" not in data
#             or len(data["choices"]) == 0
#         ):
#             print("Empty choices response:")
#             print(data)

#             if retries > 0:
#                 return openrouter_create_question(
#                     args,
#                     text,
#                     question_type,
#                     culture,
#                     question_language,
#                     retries - 1
#                 )

#             return None


#         answer = (
#             data["choices"][0]
#             ["message"]
#             ["content"]
#         )


#         if (
#             answer is None
#             or answer.strip() == ""
#         ):
#             print("Empty answer received")

#             if retries > 0:
#                 return openrouter_create_question(
#                     args,
#                     text,
#                     question_type,
#                     culture,
#                     question_language,
#                     retries - 1
#                 )

#             return None


#         print("     DONE: questions created")

#         return answer



#     except requests.exceptions.RequestException as e:

#         print("Request failed:", e)

#         if retries > 0:
#             return openrouter_create_question(
#                 args,
#                 text,
#                 question_type,
#                 culture,
#                 question_language,
#                 retries - 1
#             )

#         return None


def knowledge_preparing(args, culture, dimension, knowledge_output_dict):
    '''This function prepares the knowledge data to deliver it as knowledge_list to gerenate the question afterwards.
    It gives knowledge_list, title_list and snippet_list scaning the knowledge_path.
    '''

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
            notEnoughInfo.append(f"{dimension} in {culture}")
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

    return  knowledge_list, title_list, snippet_list

def knowledge_to_question(args, culture, dimension, knowledge_list, typ):
    '''This function separates title, snippet and knowledge from the knowledge_path and clean them. and separates them into lists to better visualization and data control to create a question.
    It gives knowledge_list, title_list and snippet_list scaning the knowledge_path.
    It creates at the end the question_reference_{timestamp}.json file with the question, options and reference answer.

    return:
    question_cleaned, abcd_options_cleaned, reference_answer, knowledge_list, title_list, snippet_list, selected_format
    '''
    output_path = f"results/questions.json"
            
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            question_vector = json.load(f)
    else:
        question_vector = {}

    if culture not in question_vector:
        question_vector[culture] = {}

    if dimension not in question_vector[culture]:
        question_vector[culture][dimension] = {}


    if args.api == "openai":
        result = openai_create_question(text=knowledge_list, question_type=typ, culture=culture, question_language=args.question_language)
    else: 
        result = interweb_create_question(args, text=knowledge_list, question_type=typ, culture=culture, question_language=args.question_language)

    if result is None:
        print("Warning: No question generated for this knowledge set.")
        selected_format = None
        question_reference = None
    else: 
        question_reference, selected_format = result

    #for each question type there is a differnet format to follow
    if question_reference is not None:

        if selected_format == "single_choice":
            question_reference = question_reference.split("Question:", 1)[1]

            parts = question_reference.split("Reference Answer:", 1)

            question_text = parts[0].replace("Question:", "").replace("Options:", "").strip()
            
            split_index = question_text.find("A)")
            if split_index == -1:
                split_index = question_text.find("a)")

            question = question_text[:split_index].strip()
            question_cleaned = question.replace("\n", " ")

            abcd_options = question_text[split_index:].strip()
            abcd_options_cleaned = abcd_options.replace("\n", " ").replace("  ", " ")

            reference_answer = parts[1].strip()

        if selected_format == "true_false":
            question_reference = question_reference.split("Question:", 1)[1]

            question_part, reference_answer = question_reference.split(
                "Reference Answer:", 1
            )

            question_text, _ = question_part.split("Options:", 1)

            question_cleaned = " ".join(question_text.split())

            abcd_options_cleaned = "NA"

            reference_answer = " ".join(reference_answer.split())
        
        if selected_format == "fill_the_blank":
            question_reference = question_reference.split("Question:", 1)[1]

            question_part, reference_answer = question_reference.split(
                "Reference Answer:", 1
            )

            question_text, _ = question_part.split("Options:", 1)

            question_cleaned = " ".join(question_text.split())

            abcd_options_cleaned = "NA"

            reference_answer = " ".join(reference_answer.split())

        if selected_format == "short_answer":

            question_reference = question_reference.split("Question:", 1)[1]

            question_part, reference_answer = question_reference.split(
                "Reference Answer:", 1
            )

            question_text, _ = question_part.split("Options:", 1)

            question_cleaned = " ".join(question_text.split())

            abcd_options_cleaned = "NA"

            reference_answer = " ".join(reference_answer.split())

        if selected_format == "long_answer":

            question_reference = question_reference.split("Question:", 1)[1]

            question_part, reference_answer = question_reference.split(
                "Reference Answer:", 1
            )

            question_text, _ = question_part.split("Options:", 1)

            question_cleaned = " ".join(question_text.split())

            abcd_options_cleaned = "NA"

            reference_answer = " ".join(reference_answer.split())
    else:
        question_cleaned = "EMPTY"
        abcd_options_cleaned = "EMPTY"
        reference_answer = "EMPTY"


    if typ not in question_vector[culture][dimension]:
        key = f"{typ} - {selected_format}"
        question_vector[culture][dimension][key] = []

    if selected_format == "single_choice":
        question_vector[culture][dimension][key].append({
            "question": question_cleaned,
            "options": abcd_options_cleaned,
            "reference_answer": reference_answer    
        })
    else:
        question_vector[culture][dimension][f"{typ} - {selected_format}"].append( {
            "question": question_cleaned,
            "reference_answer": reference_answer
        })


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question_vector, f, ensure_ascii=False, indent=2)

    return question_cleaned, abcd_options_cleaned, reference_answer, selected_format



def csv_saver(args, dimension, culture, timestamp, culture_dfs, knowledge_path, knowledge_output_dict):
    '''This function prepares the question data to deliver it in `.csv` format and save it in the `results` folder.
    return:
        None.
    '''

    knowledge_list, title_list, snippet_list= knowledge_preparing(args, culture, dimension, knowledge_output_dict)

    if args.question_type == "all":
        types = ["factual", "conceptual", "misleading", "multihop"]
    else: 
        types = [args.question_type]

    for typ in types:
        question, abcd_options, reference_answer, selected_format = knowledge_to_question(args, culture, dimension, knowledge_list, typ)

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
                final_df.to_csv(f"results/{culture}_Knowledge_QA.csv", index=False, encoding="utf-8-sig")
            else:
                print(f"Warning: No data to save for culture {culture}")
