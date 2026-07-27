from dotenv import load_dotenv
import os
import requests


ROLE_STUDENT = f"You are an expert multiple choice question solver. Your task is to analyze the question and the provided answer options carefully, determine the correct answer, and return only the letter corresponding to the selected option."

def PROMPT_STUDENT(question, options):

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
