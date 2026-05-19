from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API key initialization
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def create_question(text, question_type):
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
        Task: Answer in English.
        Instruction:
        {instruction}
        Note:
        1. The question should avoid explicitly mentioning cultural concepts, terminology,
        or characteristics, in order to effectively assess the student’s understanding
        of cultural traits.
        2. A reference answer should be provided after the question.
        Context:
        {prompt_texts}
        Question:
        Reference Answer:

        """

    # return prompt

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return content

def knowledge_to_question(knowledge_path):
    '''This function creates questions from knowledge.
       It gives knowledge_list, title_list and snippet_list scaning the knowledge_path 
    '''


    knowledge_list = []
    title_list = []
    snippet_list = []

    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item_str in data:
        item = json.loads(item_str)  # Convertir el string JSON a diccionario
        knowledge_list.append(item['Knowledge'])

        question_reference = create_question(text=knowledge_list, question_type="factual")


    with open("question_reference.json", "w", encoding="utf-8") as f:
        json.dump(question_reference, f, ensure_ascii=False, indent=2)
