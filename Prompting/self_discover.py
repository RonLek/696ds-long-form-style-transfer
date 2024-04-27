from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

def run_self_discover(source_doc, reference_doc, publication_name, reasoning_modules):
    task_description = f"Your task is to create a comprehensive style guide for {publication_name} based on the provided reference {publication_name}'s articles. The style guide should serve as a detailed reference for writers, editors, and content creators to ensure consistency and adherence to the publication's unique voice and style."
    task_instance = f'''
To create the style guide, carefully analyze the reference documents and identify the key attributes that define the {publication_name}'s style. Consider the following attributes as a starting point, but feel free to discard, modify, or add new attributes as necessary to accurately capture the {publication_name}'s style:

1. "documentStructure": [Document structure and organization]
2. "languageTone": [Language formality and tone]  
3. "vocabulary": [Vocabulary, terminology, and jargon]
4. "sentenceStructure": [Sentence structure, length, and complexity and formality]
5. "paragraphStructure": [Paragraph structure and length]
6. "grammarSyntax": [Grammar and syntax patterns]
7. "voiceUsage": [Active vs. passive voice usage]
8. "perspective": [Perspective (e.g., first person, third person)]
9. "persuasiveTechniques": [Persuasive techniques and rhetorical devices]
10. "formattingElements": [Formatting and visual elements]
11. "targetAudience": [Target audience and purpose of the content]
12. "inclusiveLanguage": [Preferred pronouns and inclusive language guidelines]
13. "punctuationCapitalization": [Punctuation and capitalization rules]
14. "citationStyle": [Acceptable sources and citation styles]
15. "writingTips": [Any specific do's and don'ts for writers to keep in mind]
..... add new attributes

For each attribute, provide a brief description and relevant examples in an array. 
Add new attributes as required by the reference document, the list isnt comprehensive.

Organize the style guide in JSON into a clear hierarchy, making it easy for LLM's to parse and reference.

The final JSON style guide should be comprehensive, enabling LLM's to internalize {publication_name}'s unique style and consistently produce content that resonates with their target audience.

Please provide the complete style guide in valid JSON format.

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