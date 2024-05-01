from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

def run_self_discover(source_doc, reference_doc, publication_name, reasoning_modules):
    task_description = f"You are an expert AI language model capable of performing style transfer from one document to another. Your task is to analyze the reference {publication_name}'s articles, identify key stylistic attributes, and then use those attributes to transfer the style of the reference document to the source document."
    task_instance = f''' To perform the style transfer, follow these steps:

Please rewrite the provided source document to match the writing style of the reference document from {publication_name} . First, analyze the reference document to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document, while preserving the key informational content of the original source document. Please provide the full rewritten document.


[SOURCE DOCUMENT]: 
{source_doc}
[REFERENCE DOCUMENT]:
{reference_doc}
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