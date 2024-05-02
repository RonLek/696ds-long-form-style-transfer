from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import csv
def attribute_scorer(tokenizer, model, attribute_index, target_text, output_text):
    (target_score, output_score) = (0, 0)
    (target_scores_list, output_scores_list) = ([], [])
    
    target_paragraphs = [paragraph.strip() for paragraph in target_text.split('\n') if paragraph.strip()]
    output_paragraphs = [paragraph.strip() for paragraph in output_text.split('\n') if paragraph.strip()]

    for paragraph in target_paragraphs:
        inputs = tokenizer(paragraph, return_tensors="pt")
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1)
    
        # Get the probabilities for attribute  class
        attribute_prob = probs[0][attribute_index].item()
    
        target_scores_list.append(attribute_prob)

    for paragraph in output_paragraphs:
        inputs = tokenizer(paragraph, return_tensors="pt")
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1)
    
        # Get the probabilities for attribute  class
        attribute_prob = probs[0][attribute_index].item()
    
        output_scores_list.append(attribute_prob)

    avg_output_score = sum(output_scores_list)/len(output_scores_list)
    avg_target_score = sum(target_scores_list)/len(target_scores_list)
#    print(avg_output_score)
#    print(avg_target_score)

    return (avg_output_score - avg_target_score) / avg_target_score


formality_tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker", max_length=512)
formality_model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
simplicity_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased", max_length=512)
simplicity_model = AutoModelForSequenceClassification.from_pretrained('/work/pi_dhruveshpate_umass_edu/project_7/classifiers/subset_simplicity_outputs_bert_large_uncased/checkpoint-7030')

for prompt in ['zero_shot', 'few_shot']:
    output_data =[['normalized_formality', 'normalized_simplicity']]
    with open('../outputs/output_tech.csv', 'r') as f_outputs:
        reader_obj = csv.reader(f_outputs, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line = next(reader_obj)
        for i in range(50): # only need 50 pairs per domain
            line = next(reader_obj)
            formality_score = attribute_scorer(formality_tokenizer, formality_model, 1, line[2], line[5])
            simplicity_score = attribute_scorer(simplicity_tokenizer, simplicity_model, 1, line[2], line[5])
            print(formality_score, simplicity_score)
            output_data.append([formality_score, simplicity_score])
    
        with open(f'{prompt}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(output_data)

        print(f'completed {prompt}')

