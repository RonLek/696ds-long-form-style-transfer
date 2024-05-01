from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json
import random

load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

def run_few_shot(source_doc, reference_docs, publication_name, use_pairs=True):
    if use_pairs:
        # Randomly select pairs of source-reference documents
        reference_pairs = random.sample(reference_docs, min(len(reference_docs), 3))  # Adjust the number of pairs as needed
        reference_pairs_str = "\n\n".join([f"SOURCE:\n{pair['source']}\n\nREFERENCE:\n{pair['reference']}" for pair in reference_pairs])
        
        # response = client.chat.completions.create(
        #     model="gpt-4-turbo-preview",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": f"You are an expert copywriter who can replicate the style of writing from one document to another. Please rewrite the provided source document to match the writing style of the reference document pairs from {publication_name}. First, analyze the reference document pairs to identify and understand the key style attributes. After gaining a high-level understanding of these style attributes, use and apply the relevant ones to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document pairs, while preserving the key informational content of the original source document. Please provide only the full rewritten document and no other auxiliary text."
        #         },
        #         {
        #             "role": "user",
        #             "content": f"SOURCE:\n{source_doc}\n\nREFERENCE PAIRS:\n{reference_pairs_str}"
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
                        "content": f"You are an expert copywriter who can replicate the style of writing from one document to another. Please rewrite the provided source document to match the writing style of the reference document pairs from {publication_name}. First, analyze the reference document pairs to identify and understand the key style attributes. After gaining a high-level understanding of these style attributes, use and apply the relevant ones to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference document pairs, while preserving the key informational content of the original source document. Please provide only the full rewritten document and no other auxiliary text."
                    },
                    {
                        "role": "user",
                        "content": f"SOURCE:\n{source_doc}\n\nREFERENCE PAIRS:\n{reference_pairs_str}"
                    }
                ]
            })
        )
    else:
        # Use a set of reference documents
        reference_docs_str = "\n\n".join(reference_docs)
        
        # response = client.chat.completions.create(
        #     model="gpt-4-turbo-preview",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": f"You are an expert copywriter who can replicate the style of writing from one document to another. Please rewrite the provided source document to match the writing style of the reference documents from {publication_name}. First, analyze the reference documents to identify and understand the key style attributes. After gaining a high-level understanding of these style attributes, use and apply the relevant ones to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference documents, while preserving the key informational content of the original source document. Please provide only the full rewritten document and no other auxiliary text."
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
                        "content": f"You are an expert copywriter who can replicate the style of writing from one document to another. Please rewrite the provided source document to match the writing style of the reference documents from {publication_name}. First, analyze the reference documents to identify and understand the key style attributes. After gaining a high-level understanding of these style attributes, use and apply the relevant ones to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.\nThe rewritten document should convincingly read as though it were written by the same author or publication as the reference documents, while preserving the key informational content of the original source document. Please provide only the full rewritten document and no other auxiliary text."
                    },
                    {
                        "role": "user",
                        "content": f"SOURCE:\n{source_doc}\n\nREFERENCE DOCUMENTS:\n{reference_docs_str}"
                    }
                ]
            })
        )
    
    return response.json()["choices"][0]["message"]["content"].strip()