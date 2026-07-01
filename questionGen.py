import re

from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API key initialization
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def create_question(text, question_type, culture, question_language):
    """
    Extracts important features and content related to a specific culture
    from a given text. Returns format: title, original snippet and knowledge extrated from the text.
    Does not invent information if there is insufficient support.
    """
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

    if question_type == "factual":
        instruction = f"""
        Based on the context, think through all relevant cultural points step by step and generate a factual question. The question type can include single-choice, true/false, or fill-in-the-blank. Ensure that the question stem is clear, the options are plausible but misleading (distractors), and the answer is accurate.
        """
    elif question_type == "conceptual":
        instruction = f"""
        Based on the context, think through all relevant cultural points step by step and generate a conceptual explanation question. The question should focus on the learner’s understanding of the concepts, structures, or values inside cultural phenomena, rather than simple memorization. Suitable formats include multiple-choice or true/false questions. Ensure the question is thought-provoking and the answer is well-justified.        
        """  
    elif question_type == "misleading":
        instruction = f"""
        Based on the context, think through all relevant cultural points step by step and generate a misleading question to assess whether learners can identify cultural misunderstandings, stereotypes, or biases. The question should focus on learners’ critical thinking about culture, identifying which statements or Behaviors reflect misunderstandings, oversimplifications, biases, or stereotypes, and guide them toward more accurate or respectful understandings. Possible formats include multiple-choice, true/false, case analysis, or short-answer questions.
        """
    elif question_type == "multihop":
        instruction = f"""
        and generate a multi-hop reasoning question to assess whether the learner can synthesize multiple cultural elements and understand the deeper logic or internal connections among cultural phenomena. The question should prompt learners to start from multiple information points, integrate cultural knowledge, and perform logical analysis, comparison, or generalization. Scenario-based, integrated analysis, or comparative reasoning questions are recommended.
        """
    else:
        raise ValueError("Invalid question type. Must be one of: factual, comparative, causal, hypothetical.")


    prompt = f"""
        Task: Answer in {idiom}.
        Instruction:
        {instruction}
        Note:
        1. The question should avoid explicitly mentioning cultural concepts, terminology,
        or characteristics, in order to effectively assess the student’s understanding
        of cultural traits.
        2. A reference answer should be provided after the question.
        3. Do not change the structure of "Question", "Options" and "Reference Answer" in the output, as they will be used for evaluation. Just fill in the content after these labels.
        Context:
        {prompt_texts}
        Give question and options based on the context provided. The student does not have access to the context, so the question should be answerable without the context.
        (example: Question? A) option 1, B) option 2, C) option 3, D) option 4)
        Reference Answer: -. 
        """

    # return prompt

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content

def knowledge_to_question(knowledge_path, culture, dimension, timestamp, question_language, notEnoughInfo_dimension):
    '''This function creates questions from knowledge.
       It gives knowledge_list, title_list and snippet_list scaning the knowledge_path

       return:
        question_cleaned, abcd_options_cleaned, reference_answer, knowledge_list, title_list, snippet_list
    '''
    output_path = f"results/question_reference_{timestamp}.json"
    

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            question_vector = json.load(f)
    else:
        question_vector = {}

    knowledge_list = []
    title_list = []
    snippet_list = []

    with open(knowledge_path, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)
    
    knowledge_items = knowledge_data[culture][dimension]

    print(f"To question: Processing knowledge item: {dimension} in {culture}") 

    for knowledge_set in knowledge_items:

        if not knowledge_set or not knowledge_set.strip():
            print("Warning: empty knowledge_set, skipping")
            continue
        
        clean_json = re.sub(r"^```json\s*|\s*```$", "", knowledge_set, flags=re.DOTALL).strip()

        try:
            item = json.loads(knowledge_set)  # it converts the string back to a dictionary 

        except json.JSONDecodeError:
            print(f"Warning: invalid JSON, skipping: {knowledge_set[:50]}...")
            continue

        
        know = item.get('Knowledge') or item.get('knowledge')

        if know is None:
            know_cleaned = ""
        else:
            know_cleaned = know.replace(";", ",").strip()

        titl = item.get('Title') or item.get('title')
        title_cleaned = titl.replace(";", ",").strip()

        snipp = item.get('Snippet') or item.get('snippet')
        snippet_cleaned = snipp.replace(";", ",").strip()

        if "(info missing)" in title_cleaned:
            notEnoughInfo_dimension.append(f"{dimension} in {culture}")
        else:
            knowledge_list.append(know_cleaned)
            title_list.append(title_cleaned)
            snippet_list.append(snippet_cleaned)
            
    if notEnoughInfo_dimension is not None:
        print("notEnoughInfo_dimension:", notEnoughInfo_dimension)  #this should be empty if all is working
        print("this dimensions are going to be run again with another query. FIND THR SOURCE OF THE PROBLEM")

    question_reference = create_question(text=knowledge_list, question_type="factual", culture=culture, question_language=question_language)
    
    parts = question_reference.split("Reference Answer:")

    question_text = parts[0].replace("Question:", "").strip()
    
    split_index = question_text.find("A)")
    if split_index == -1:
        split_index = question_text.find("a)")

    question = question_text[:split_index].strip()
    question_cleaned = question.replace("\n", " ")

    abcd_options = question_text[split_index:].strip()
    # abcd_options_cleaned = abcd_options.replace("  ", " ")
    # abcd_options_cleaned = abcd_options.replace("\n", " ")
    abcd_options_cleaned = abcd_options.replace("\n", " ").replace("  ", " ")

    reference_answer = parts[1].strip()

    if culture not in question_vector:
        question_vector[culture] = {}

    question_vector[culture][dimension] = {
        "question": question_cleaned,
        "options": abcd_options_cleaned,
        "reference_answer": reference_answer
    }


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question_vector, f, ensure_ascii=False, indent=2)

    return question_cleaned, abcd_options_cleaned, reference_answer, knowledge_list, title_list, snippet_list



def csv_saver(args, dimension, culture, timestamp, culture_dfs, knowledge_path):

    
    notEnoughInfo_dimension = []

    question, abcd_options, reference_answer, knowledge_list, title_list, snippet_list = knowledge_to_question(knowledge_path, culture, dimension, timestamp, args.question_language, notEnoughInfo_dimension)

    output_path = f"results/knowledge_output_{timestamp}.json"

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

    knowledge_vector[culture][dimension] = {
        "data": data_knowledge_info
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_vector, f, ensure_ascii=False, indent=2)


    df_questions_reference = pd.DataFrame({
        "question": [question],
        "abcd_options": [abcd_options],
        "reference_answer": [reference_answer]
    })



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
