import time
import os
from googleapiclient import model
import requests
import random

from dotenv import load_dotenv

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
    - Does the answer contain correct information?
    - Are there any factual errors?

    2. Completeness:
    - Does the answer cover the main points from the reference answer?
    - Is important information missing?

    3. Relevance:
    - Does the answer directly respond to the question?
    - Does it avoid unrelated information?

    Decision rules:
    - PASS: The answer is correct and sufficiently complete. Minor wording differences are acceptable.
    - FAIL: The answer is incorrect, irrelevant, or missing important information.

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


def interweb_student(llm_model, question, options, retries = 5):
    """
    LLM takes the rol from a student and it answers the question given to it.
    """

    load_dotenv()
    INTERWEB_API_KEY = os.getenv("INTERWEB_API_KEY")

    url = "https://interweb.l3s.uni-hannover.de"

    headers = {
        "Authorization": f"Bearer {INTERWEB_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json" 
    }

    user_prompt = PROMPT_STUDENT(question, options)

    payload = {
        "model": llm_model,  # Replace with the model available in your API. gpt-4o-mini
        "messages": [
            {
                "role": "system",
                "content": ROLE_STUDENT,
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n"
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
                return interweb_student(
                    llm_model,
                    question,
                    options,
                    retries - 1
                )
            
            return None
        
        answer = response.json()["choices"][0]["message"]["content"]

        if answer is None or "[]" in answer or answer.strip() == "":
            if retries > 0:
                print(f"Empty response. Retrying... attempts left: {retries}")
                return interweb_student(
                    llm_model,
                    question,
                    options,
                    retries - 1
                )

            return None

        return answer
    
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

        if retries > 0:
            return interweb_student(
                llm_model,
                question,
                options,
                retries - 1
            )

        return None



def openrouter_student(llm_model, question, options, question_format, retries = 5):
    """
    LLM takes the rol from a student and it answers the question given to it using Openrouter.
    """

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    user_prompt = PROMPT_STUDENT(question, options, question_format)

    data = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": ROLE_STUDENT,
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n"
            }
        ]
    }

    for attempt in range(retries):
 
        response = requests.post(
            f"{url}/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code == 200:

            answer = response.json()["choices"][0]["message"]["content"]
            return answer

        elif response.status_code == 429:

            wait_time = (2 ** attempt) + random.random()

            print(
                f"429 Rate limit {llm_model}. "
                f"Retry {attempt+1}/{retries}. "
                f"Waiting {wait_time:.2f}s"
            )

            time.sleep(wait_time)

        else: 
            response.raise_for_status()

    raise RuntimeError(
    f"{llm_model} failed after {retries} retries"

    )


def openrouter_grader(question, reference_answer, llm_answer, question_type, retries = 5):
    """
    LLM takes the rol from a student and it answers the question given to it using Openrouter.
    """

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    user_prompt = PROMPT_GRADER(question, reference_answer, llm_answer, question_type)

    data = {
        "model": "minimax/minimax-m3:free",
        "messages": [
            {
                "role": "system",
                "content": ROLE_GRADER,
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n"
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


    return answer