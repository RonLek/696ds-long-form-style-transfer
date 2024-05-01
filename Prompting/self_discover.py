from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

def run_self_discover(source_doc, reference_docs, publication_name, reasoning_modules):
    reference_docs_str = "\n\n".join(reference_docs)
    task_description = f"You are an expert AI language model capable of performing style transfer from one document to another. Your task is to analyze the reference {publication_name}'s articles, identify key stylistic attributes, and then use those attributes to transfer the style of the reference document to the source document."
    task_instance = f''' To perform the style transfer, follow these steps:

Analyze the reference documents and identify the key attributes that define the {publication_name}'s style.
Internally imagine a comprehensive style guide based on the identified attributes, making it easy for you to parse and reference.
Using the internal style guide you created, transfer the style of the reference document to the source document. Ensure that the resulting content consistently reflects the {publication_name}'s unique voice and style while preserving the original meaning and information of the source document.
Provide only the complete style transfer result.



[SOURCE DOCUMENT]: 
{source_doc}
[REFERENCE DOCUMENT]:
{reference_docs_str}
'''
    selected_modules = select_reasoning_modules(task_description, reasoning_modules)
    adapted_modules = adapt_reasoning_modules(selected_modules, task_instance)
    reasoning_structure = implement_reasoning_structure(adapted_modules, task_description)
    result = execute_reasoning_structure(reasoning_structure, task_instance)
    return result

def select_reasoning_modules(task_description, reasoning_modules):
    prompt = f"Given the task: {task_description}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n" + "\n".join(reasoning_modules)
    selected_modules = query_openai(prompt)
    return selected_modules

def adapt_reasoning_modules(selected_modules, task_example):
    prompt = f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task_example}"
    adapted_modules = query_openai(prompt)
    return adapted_modules

def implement_reasoning_structure(adapted_modules, task_description):
    prompt = f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_description}"
    reasoning_structure = query_openai(prompt)
    return reasoning_structure

def execute_reasoning_structure(reasoning_structure, task_instance):
    prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task_instance}"
    solution = query_openai(prompt)
    return solution

def query_openai(prompt, temperature=0.1):
    # while True:
    #     try:
    #         response = client.chat.completions.create(
    #             model=os.environ["MODEL"],
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=temperature,
    #             n=1,
    #         )
    #         content = response.choices[0].message.content.strip()
    #         return content
    #     except Exception as e:
    #         print("Failure querying the AI. Retrying...")
    #         time.sleep(1)
    while True:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}"
                },
                data=json.dumps({
                    "model": "meta-llama/llama-3-70b-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "n": 1
                })
            )
            content = response.json()["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            print("Failure querying the AI. Retrying...")
            time.sleep(1)