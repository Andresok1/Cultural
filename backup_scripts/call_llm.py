import os
import re
from urllib import response
from ddgs.http_client2 import random
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from time import perf_counter
import time
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


ROLE_KNOWLEDGE = f"You are an expert assistant in cultural text analysis. Your task is to read the following texts provided and extract only the relevant information related to the culture."

def PROMPT_KNOWLEDGE(culture, dimension, prompt_texts):
    user_prompt = f"""
        You will be provided with a collection of documents related to a specific culture and dimension.

        Your task is to analyze all documents together as a single knowledge source and identify the most relevant pieces of information related to the given culture and dimension.

        Extract all distinct knowledge entries supported by relevant evidence in the provided documents. Do not require a minimum number of entries. Only create an entry when the documents contain meaningful information directly related to the requested dimension and culture.
        Relevant evidence describes a practice, expectation, behavior, or fact concerning the requested dimension in the requested cultural context. Mentioning the culture alone, or discussing an adjacent topic without an explicit connection to the dimension, is not sufficient.
        A finding may address only one specific aspect of the dimension; it does not need to describe the entire culture. Preserve any limits stated in the source, such as location, setting, group, or uncertainty. Do not turn a specific example into a general cultural rule.

        For each relevant finding, return:

        - title: A short title summarizing the main idea or cultural feature.
        - culture: {culture}
        - dimension: {dimension}
        - snippet: The exact original text extracted from the provided documents that supports this finding. Do not modify, translate, summarize, or paraphrase the snippet. Preserve the original language and wording.
        - knowledge: A concise summary in English explaining what can be learned from this snippet about the {dimension} dimension in {culture} culture. The knowledge summary must be based exclusively on the provided snippet.

        Important rules:
        - Treat all provided documents as one combined source. Do not analyze documents independently.
        - Search across the entire document collection to identify distinct findings. When multiple documents describe the same cultural feature, avoid duplicate entries and select a supporting excerpt from one document for that finding.
        - Each snippet must be one continuous excerpt with enough context to support the knowledge summary. Do not stitch passages together or infer explanations that the excerpt does not provide.
        - Each snippet and knowledge pair must represent one distinct and meaningful cultural finding.
        - If multiple unrelated important findings exist, create separate entries for each one.
        - Do not create redundant entries describing the same information.
        - Do not add external knowledge or assumptions beyond the provided documents.
        - If no relevant information is found for the requested culture and dimension, write as title: (NOT RELEVANT INFORMATION) and the rest as "EMPTY" .

        Return the output only in the following structured JSON format:

        [
            {{
                "title": "...",
                "culture": "{culture}",
                "dimension": "{dimension}",
                "snippet": "...",
                "knowledge": "..."
            }}
        ]
        Documents:
        {prompt_texts}

        Do not introduce your answer with any text or explanation. Only return the JSON array of knowledge entries.
        """
    return user_prompt

ROLE_QUESTION = """
You are an expert educational assessment designer specialized in cultural knowledge evaluation.
Your task is to create accurate assessment questions from provided cultural information.
You design items that evaluate understanding, reasoning, and interpretation.
Questions may include multiple-choice, true/false, short-answer, or long-answer formats.
"""

QUESTION_TYPES = {
    "factual": "Based on the context, think through all relevant cultural points step by step and generate a factual question. The question type can include single-choice, true/false, or fill-in-the-blank. Ensure that the question stem is clear, the options are plausible but misleading (distractors), and the answer is accurate.",
    "conceptual": "Based on the context, think through all relevant cultural points step by step and generate a conceptual explanation question. The question should focus on the learner’s understanding of the concepts, structures, or values inside cultural phenomena, rather than simple memorization. Suitable formats include multiple-choice or true/false questions. Ensure the question is thought-provoking and the answer is well-justified.",
    "misleading": "Based on the context, think through all relevant cultural points step by step and generate a misleading question to assess whether learners can identify cultural misunderstandings, stereotypes, or biases. The question should focus on learners’ critical thinking about culture, identifying which statements or Behaviors reflect misunderstandings, oversimplifications, biases, or stereotypes, and guide them toward more accurate or respectful understandings. Possible formats include multiple-choice, true/false, case analysis, or short-answer questions.",
    "long_form": "Based on the context, think through all relevant cultural points step by step and generate a multi-hop reasoning question to assess whether the learner can synthesize multiple cultural elements and understand the deeper logic or internal connections among cultural phenomena. The question should prompt learners to start from multiple information points, integrate cultural knowledge, and perform logical analysis, comparison, or generalization. Scenario-based, integrated analysis, or comparative reasoning questions are recommended."
}



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

        "long_form": 
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



ROLE_STUDENT = f"""You are an expert question solver. Your task is to carefully analyze the provided question and determine the most accurate answer based on your knowledge and reasoning.

You must follow the instructions provided for each question type and return only the requested answer format.

Do not include unnecessary explanations, introductions, conclusions, or additional formatting unless explicitly requested by the question instructions."""

ROLE_GRADER = f"""You are an expert evaluator responsible for grading answers to academic questions.

Your task is to compare a student's answer with the reference answer and determine whether the student's response is correct.

Evaluate answers based on:
- Accuracy: Is the answer factually correct?
- Completeness: Does the answer include the essential information required?
- Relevance: Does the answer directly address the question?

You must be objective and consistent. Do not judge based on wording differences if the meaning is correct."""

def PROMPT_GRADER(question, reference_answer, llm_answer, question_type):

    if question_type == "fill_the_blank":
        type_criteria = """
        For fill-in-the-blank questions:
        The student does not need to reproduce the complete reference answer.
        PASS if the student provides the essential information required to correctly
        complete the sentence.

        Examples, explanations, or additional details in the reference answer are
        not required unless they are necessary for the meaning of the answer.
        """

    elif question_type == "short_answer":
        type_criteria = """
        For short-answer questions:
        PASS if the student provides the main information required to answer the question.
        Minor details or examples from the reference answer may be omitted if the core
        answer remains correct.
        """

    elif question_type == "long_answer":
        type_criteria = """
        For long-answer questions:
        Evaluate both correctness and completeness.
        The student should cover the main ideas required by the question, but does not
        need to reproduce the reference answer word for word.
        """

    else:
        raise ValueError(f"Unsupported question type: {question_type}")

    prompt = f"""
    You are grading a student's response.

    Question type:
    {question_type}

    Question:
    {question}

    Reference answer:
    {reference_answer}

    Student answer:
    {llm_answer}

    Evaluate the student answer using these criteria:

    1. Accuracy:
    Does the answer contain correct information?
    Are there any factual errors or contradictions?

    2. Relevance:
    Does the answer directly respond to the question?
    Does it avoid unrelated information?

    3. Required information:
    Determine which information from the reference answer is actually necessary
    to correctly answer the question.

    {type_criteria}

    Decision rules:

    PASS:
    The answer is correct and contains the information required for this question type.

    FAIL:
    The answer is incorrect, contradictory, irrelevant, or missing essential information.

    Do not judge based on identical wording or lexical similarity.

    Return only one word:
    PASS or FAIL.
    """

    return prompt

def PROMPT_STUDENT(question, options, question_format):

    if question_format == "single_choice":
        prompt = f"""
        You will receive a multiple choice question with four possible answers labeled A, B, C, and D.

        Your instructions:

        Read the question carefully.
        Evaluate each option and identify the best answer.
        Respond with only the letter of the correct option.
        Do not include explanations, reasoning, punctuation, additional words, or formatting.
        Your output must be exactly one character: A, B, C, or D.

        Example input:
            Question: What is the capital of France?
            A) Berlin
            B) Madrid
            C) Paris
            D) Rome

            Example output:
            C

        Now solve the following question and provide only the answer letter:

        Question:
        {question}

        Options:
        {options}

        """
    elif question_format == "fill_the_blank":
        prompt = f"""
        You will receive a fill-in-the-blank question.

        Your instructions:

        Read the question carefully.
        Determine the correct word or phrase to fill in the blank.
        Respond with only the answer, without any additional words, punctuation, or formatting.

        Example input:
            Question: The capital of France is _______.

            Example output:
            Paris

        Now solve the following question and provide only the answer:

        Question:
        {question}
        """
    elif question_format == "true_false":
        prompt = f"""
        You will receive a true/false question.

        Your instructions:

        Read the question carefully.
        Determine whether the statement is true or false.
        Respond with only "True" or "False", without any additional words, punctuation, or formatting.

        Example input:
            Question: The Earth is flat.

            Example output:
            False

        Now solve the following question and provide only the answer:

        Question:
        {question}
        """
    elif question_format == "short_answer":
        prompt = f"""
        You will receive a short answer question.

        Your instructions:

        Read the question carefully.
        Provide a concise and accurate answer. Write a short essay answering the following question. Expected answer length: 3-5 sentences.

        Respond with only the answer, without any additional words, punctuation, or formatting.

        Example input:
            Question: What is the process of photosynthesis?

            Example output:
            Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make their own food. During this process, chlorophyll in plant cells captures energy from sunlight and helps convert the raw materials into glucose, a type of sugar. Oxygen is released as a byproduct of photosynthesis and is given off into the atmosphere. This process is important because it provides plants with energy and supplies oxygen needed by many living organisms.


        Now solve the following question and provide only the answer:

        Question:
        {question}
        """
    elif question_format == "long_answer":
        prompt = f"""
        You will receive a long answer question.

        Your instructions:

        Read the question carefully.
        Provide a detailed and comprehensive answer. Write a long essay answering the following question. Expected answer length: 5-8 sentences.

        Respond with only the answer, without any additional words, punctuation, or formatting.

        Example input:
            Question: Explain the theory of evolution by natural selection.

            Example output:
            The theory of evolution by natural selection, proposed by Charles Darwin, explains how species change over time through a process of adaptation to their environment. According to this theory, individuals within a species exhibit variations in traits, some of which may provide advantages for survival and reproduction. Those individuals with favorable traits are more likely to survive and pass on their genes to the next generation. Over many generations, these advantageous traits become more common within the population, leading to evolutionary changes. Natural selection acts on the genetic diversity present in populations, driving the development of new species and contributing to the diversity of life on Earth. This process is influenced by environmental pressures, competition for resources, and random genetic mutations that can introduce new traits. The theory of evolution by natural selection is supported by extensive evidence from fossil records, comparative anatomy, and molecular biology.
        
        Now solve the following question and provide only the answer:

        Question:
        {question}
        """ 
    else: 
        raise ValueError(f"Unsupported question type: {question_format}")
    
    return prompt












def call_llm(
    endpoint,
    llm_model,
    prompt,
    role,
    text=None,
    culture=None,
    dimension=None,
    question_type = None,
    language=None,
    question=None,
    options=None,
    question_format=None,
    reference_answer=None,
    llm_answer=None,
    retries=2,
    status_label="answering"
):
    """
    Generic LLM caller for OpenAI, OpenRouter, Interweb and Inference endpoints.
    Only the prompt/model changes when calling this function.
    """

    print("This is the allmighty")
    if role == "ROLE_KNOWLEDGE":
        if not text:
            print(f"No documents were given to produce a knowledge entry.")
            return None

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        prompt_texts = "\n\n".join([f"Text {i}:\n{text}" for i, text in enumerate(texts, 1)])

        user_prompt = PROMPT_KNOWLEDGE(culture, dimension, prompt_texts)

    elif role == "ROLE_STUDENT":
        user_prompt = PROMPT_STUDENT(question, options, question_format)

    elif role == "ROLE_GRADER":
        user_prompt = PROMPT_GRADER(question, reference_answer, llm_answer, question_type)

    elif role == "ROLE_QUESTION":
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

    load_dotenv()

    configs = {

        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            }
        },

        "openrouter": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
            }
        },

        "interweb": {
            "url": "https://interweb.l3s.uni-hannover.de/v1/chat/completions",
            "api_key": os.getenv("INTERWEB_API_KEY"),
            "headers": {
                "Authorization": f"Bearer {os.getenv('INTERWEB_API_KEY')}",
                "accept": "application/json",
                "Content-Type": "application/json"
            }
        },

        "inference": {
            "url": "https://inference.kbs.uni-hannover.de/v1",
            "api_key": os.getenv("INFERENCE_API_KEY")
        }
    }


    config = configs[endpoint]


    for attempt in range(retries + 1):

        try:

            if role == "ROLE_KNOWLEDGE":
                print(
                    f"{language + ' | ' if language else ''}"
                    f"{endpoint} / {llm_model} | "
                    f"Sending request "
                    f"(attempt {attempt + 1}/{retries + 1})...",
                    flush=True
                )


            started = perf_counter()


            # OpenAI compatible HTTP endpoints
            if endpoint in ["openai", "openrouter", "interweb"]:


                payload = {
                    "model": llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": role
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }


                response = requests.post(
                    config["url"],
                    headers=config["headers"],
                    json=payload,
                    timeout=60
                )


                if response.status_code == 429:

                    wait_time = min(2 ** attempt, 60)

                    print(
                        f"Rate limit. Waiting {wait_time}s..."
                    )

                    time.sleep(wait_time)
                    continue


                if response.status_code != 200:

                    print(
                        response.status_code,
                        response.text
                    )

                    continue


                answer = response.json()["choices"][0]["message"]["content"]



            # Inference endpoint using OpenAI client
            elif endpoint == "inference":


                client = OpenAI(
                    base_url="https://inference.kbs.uni-hannover.de/v1",
                    api_key=config["api_key"],
                    timeout=60
                )


                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": role
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


                answer = response.choices[0].message.content



            else:

                raise ValueError(
                    f"Unknown endpoint: {endpoint}"
                )



            if (
                answer is None
                or answer.strip() == ""
                or "[]" in answer
            ):

                raise ValueError(
                    "Empty response"
                )


            print(
                f"{'ANSWERING':<9} | "
                f"{status_label:<10} | "
                f"{endpoint} / {llm_model} | "
                f"Finished after {perf_counter()-started:.1f}s"
            )

            if role == "ROLE_QUESTION":
                return answer, selected_format
            else: 
                return answer

        except (
            requests.exceptions.RequestException,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            ValueError
        ) as e:


            print(
                f"Error attempt {attempt+1}/{retries+1}: "
                f"{type(e).__name__}: {e}"
            )


            if attempt < retries:

                wait_time = min(15 * (2 ** attempt), 60)

                print(
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

            else:

                print(
                    f"Game Over: {endpoint} / {llm_model}"
                )

                return None
