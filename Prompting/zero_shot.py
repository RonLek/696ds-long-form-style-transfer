from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# def run_zero_shot(source_doc, reference_doc, publication_name):
#     # response = client.chat.completions.create(
#     #     model="gpt-4-turbo-preview",
#     #     messages=[
#     #         {
#     #             "role": "system",
#     #             "content": f"Please rewrite the provided source document to match the writing style of the reference document from {publication_name}. First, analyze the reference document to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document, while preserving the key informational content of the original source document. Please provide the full rewritten document."
#     #         },
#     #         {
#     #             "role": "user",
#     #             "content": f"SOURCE:\n{source_doc}\n\nREFERENCE:\n{reference_doc}"
#     #         }
#     #     ],
#     #     temperature=0.1,
#     #     max_tokens=4095,
#     #     top_p=1,
#     #     frequency_penalty=0,
#     #     presence_penalty=0
#     # )
#     response = requests.post(
#         url="https://openrouter.ai/api/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}"
#         },
#         data=json.dumps({
#             "model": "meta-llama/llama-3-70b-instruct",
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": f"You are an expert copywriter who can replicate the style of writing from one document to another. Please rewrite the provided source document to match the writing style of the reference document from {publication_name}. First, analyze the reference document to identify and understand the key style attributes, After gaining a high-level understanding of these style attributes, Use and apply the relevant ones to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document, while preserving the key informational content of the original source document. Please provide only the full rewritten document and no other auxillary text."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"SOURCE:\n{source_doc}\n\nREFERENCE:\n{reference_doc}"
#                 }
#             ]
#         })
#     )
#     return response.json()["choices"][0]["message"]["content"].strip()

def run_zero_shot(source_doc, reference_doc, publication_name, use_publication_name=True, use_reference_doc=True):
    if use_publication_name and use_reference_doc:
        prompt = f"Please rewrite the provided source document to match the writing style of the reference document from {publication_name}. First, analyze the reference document to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document, while preserving the key informational content of the original source document. Please provide the full rewritten document."
        content = f"SOURCE:\n{source_doc}\n\nREFERENCE:\n{reference_doc}"
    elif use_publication_name:
        prompt = f"Please rewrite the provided source document to match the writing style of {publication_name}. First, analyze the source document to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as {publication_name}, while preserving the key informational content of the original source document. Please provide the full rewritten document."
        content = f"SOURCE:\n{source_doc}"
    else:
        prompt = f"Please rewrite the provided source document to match the writing style of the reference document. First, analyze the reference document to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document, while preserving the key informational content of the original source document. Please provide the full rewritten document."
        content = f"SOURCE:\n{source_doc}\n\nREFERENCE:\n{reference_doc}"
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        },
        data=json.dumps({
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        })
    )
    return response.json()["choices"][0]["message"]["content"].strip()