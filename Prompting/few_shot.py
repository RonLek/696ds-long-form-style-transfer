from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

def run_few_shot(source_doc, reference_docs, publication_name):
    reference_docs_str = ""
    skipped_docs = []
    for doc in reference_docs:
        if isinstance(doc, str):
            reference_docs_str += doc + "\n\n"
        else:
            skipped_docs.append(doc)
    
    if skipped_docs:
        print(f"Skipped {len(skipped_docs)} document(s) due to non-string type in few-shot prompting.")

    # response = client.chat.completions.create(
    #     model="gpt-4-turbo-preview",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": f"Please rewrite the provided source document to match the writing style of the reference documents from {publication_name}. First, analyze the reference documents to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference documents, while preserving the key informational content of the original source document. Please provide the full rewritten document."
    #         },
    #         {
    #             "role": "user",
    #             "content": f"SOURCE:\n{source_doc}\n\nREFERENCE DOCUMENTS:\n{reference_docs_str}"
    #         }
    #     ],
    #     temperature=0.1,
    #     max_tokens=4095,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        },
        data=json.dumps({
            "model": "meta-llama/llama-3-70b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": f"Please rewrite the provided source document to match the writing style of the reference documents from {publication_name}. First, analyze the reference documents to identify and understand the key style attributes, such as:\n1. Overall document structure and organization\n2. Formality and tone of language\n3. Vocabulary, terminology, and jargon typical of the specific domain\n4. Sentence structure, length, and complexity\n5. Paragraph structure and length\n6. Grammatical and syntactical patterns\n7. Use of active vs. passive voice\n8. Perspective (e.g., first person, third person)\n9. Persuasive techniques and rhetorical devices employed\n10. Formatting elements like headings, bullets, emphasis\nAfter gaining a high-level understanding of these style attributes . Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference documents, while preserving the key informational content of the original source document. Please provide only the full rewritten document. Do not return anything else other than the full rewritten document"
                },
                {
                    "role": "user",
                    "content": f"SOURCE:\n{source_doc}\n\nREFERENCE DOCUMENTS:\n{reference_docs_str}"

                }
            ]
        })
    )
    return response.json()["choices"][0]["message"]["content"].strip()