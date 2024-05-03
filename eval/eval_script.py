from transformers import AutoTokenizer, AutoModelForSequenceClassification, LongformerTokenizerFast, LongformerForSequenceClassification
import torch
from torch.nn import functional as F
from evaluate import load
import csv

def bertscorer(target_text, output_text):
    if not len(target_text) or not len(output_text):
        return [0] * 3

    output_paragraphs = [paragraph.strip() for paragraph in output_text.split('\n') if paragraph.strip()]
    output_text = '\n'.join(output_paragraphs[1:]) # skips the "here's rewritten... part"

    results = bertscore.compute(predictions=[output_text], references=[target_text], model_type="allenai/longformer-large-4096", lang="en")
    return [results['precision'][0], results['recall'][0], results['f1'][0]]

def publication_scorer(output_text):
    if not len(output_text):
        return [0] * len(publication_list)
    output_paragraphs = [paragraph.strip() for paragraph in output_text.split('\n') if paragraph.strip()]
    output_text = '\n'.join(output_paragraphs[1:]) # skips the "here's rewritten... part"

    inputs = publication_tokenizer(output_text, return_tensors='pt')
    with torch.no_grad():
        outputs = publication_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    return probs.tolist()[0]

def attribute_scorer(tokenizer, model, attribute_index, target_text, output_text):
    (target_score, output_score) = (0, 0)
    (target_scores_list, output_scores_list) = ([], [])
    
    target_paragraphs = [paragraph.strip() for paragraph in target_text.split('\n') if paragraph.strip()]
    output_paragraphs = [paragraph.strip() for paragraph in output_text.split('\n') if paragraph.strip()]

    if not len(target_paragraphs) or not len(output_paragraphs):
        return 0
    #print(target_paragraphs[0])
    print(output_paragraphs[1])

    for paragraph in target_paragraphs:
        inputs = tokenizer(paragraph, return_tensors="pt")
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1)
    
        # Get the probabilities for attribute  class
        attribute_prob = probs[0][attribute_index].item()
    
        target_scores_list.append(attribute_prob)

    for paragraph in output_paragraphs[1:]:
        inputs = tokenizer(paragraph, return_tensors="pt")
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1)
    
        # Get the probabilities for attribute  class
        attribute_prob = probs[0][attribute_index].item()
    
        output_scores_list.append(attribute_prob)

    if not len(target_scores_list):
        print(target_paragraphs)
        return 0

    if not len(output_scores_list):
        print(output_paragraphs)
        return 0

    avg_output_score = sum(output_scores_list)/len(output_scores_list)
    avg_target_score = sum(target_scores_list)/len(target_scores_list)
#    print(avg_output_score)
#    print(avg_target_score)

    return (avg_output_score - avg_target_score) / avg_target_score


formality_tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker", max_length=512)
formality_model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
simplicity_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased", max_length=512)
simplicity_model = AutoModelForSequenceClassification.from_pretrained('/work/pi_dhruveshpate_umass_edu/project_7/classifiers/subset_simplicity_outputs_bert_large_uncased/checkpoint-7030')
bertscore = load("bertscore")
publication_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=1024)
publication_model = LongformerForSequenceClassification.from_pretrained('/work/pi_dhruveshpate_umass_edu/project_7/classifiers/tech/longformer-finetuned-tech', num_labels = 6)
publication_list = ['cnet', 'engadget', 'wired', 'techcrunch', 'theverge', 'arstechnica']

for prompt_index, prompt in enumerate(['zero_shot', 'few_shot']):
    output_data =[['normalized_formality', 'normalized_simplicity', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'] + publication_list]
    with open('../outputs/output_tech.csv', 'r') as f_outputs:
        reader_obj = csv.reader(f_outputs, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line = next(reader_obj)
        for _ in range(5): # only need 50 pairs per domain
            line = next(reader_obj)
            formality_score = attribute_scorer(formality_tokenizer, formality_model, 1, line[2], line[5 + prompt_index])
            simplicity_score = attribute_scorer(simplicity_tokenizer, simplicity_model, 1, line[2], line[5 + prompt_index])
            bscores = bertscorer(line[2], line[5 + prompt_index])
            publication_scores = publication_scorer(line[5 + prompt_index])
            print(formality_score, simplicity_score, bscores, publication_scores)
            row_result = [formality_score, simplicity_score] + bscores + publication_scores
            print(row_result)
            output_data.append(row_result)
    
        with open(f'{prompt}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(output_data)

        print(f'completed {prompt}')

