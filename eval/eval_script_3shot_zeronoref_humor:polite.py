from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import csv
from evaluate import load
import pandas as pd


def attribute_scorer(tokenizer, model, attribute_index, target_text, output_text,pair_index):
    (target_score, output_score) = (0, 0)
    (target_scores_list, output_scores_list) = ([], [])
    
    target_paragraphs = [paragraph.strip() for paragraph in target_text.split('\n') if paragraph.strip()]
    output_paragraphs = [paragraph.strip() for paragraph in output_text.split('\n') if paragraph.strip()]

    if not len(target_paragraphs) or not len(output_paragraphs):
        return 0
        # print(target_paragraphs[0])
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

    for paragraph in output_paragraphs:
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
    
    if pair_index in [17, 25, 32, 80, 89]:
        print("op score", avg_output_score)
        print("t score", avg_target_score)

    return abs(avg_output_score - avg_target_score) / avg_target_score


humor_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", max_length=512)
humor_model = AutoModelForSequenceClassification.from_pretrained("/work/pi_dhruveshpate_umass_edu/dchiplonker_umass_edu/humor/humor_outputs_bert_base_uncased_1e-5_E2_B8/checkpoint-3150")
politeness_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", max_length=512)
politeness_model = AutoModelForSequenceClassification.from_pretrained('/work/pi_dhruveshpate_umass_edu/dchiplonker_umass_edu/polite/politeness_outputs_bert_base_uncased_100k/checkpoint-45000')

dataset_benchmark = pd.read_csv('dataset.csv') #datset discord
# for prompt in ['zero_shot', 'few_shot']:
# for prompt_index, prompt in enumerate(['zero_shot', 'few_shot', 'self_discover_absolute']):
for prompt_index, prompt in enumerate(['zero_shot_no_ref', '3shot_few_shot']):
    output_data =[['normalized_humor', 'normalized_politeness']]
    with open('output3shot_zeronoref.csv', 'r') as f_outputs: #combined dataset
        reader_obj = csv.reader(f_outputs, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line = next(reader_obj)
        for pair_index in range(100):  # 100 pairs, 20 per domain
            line = next(reader_obj)
            try:
                humor_score = attribute_scorer(humor_tokenizer, humor_model, 1, dataset_benchmark.iloc[pair_index]['content2'], line[5 + prompt_index],pair_index)
                politeness_score = attribute_scorer(politeness_tokenizer, politeness_model, 1, dataset_benchmark.iloc[pair_index]['content2'], line[5 +
                                                                                                         prompt_index],pair_index)
                print(humor_score, politeness_score)
                output_data.append([humor_score, politeness_score])
            except Exception as e:
                print(f'*** error in line {pair_index} ***')
                print(e)
                output_data.append([0] * 2)
    
        with open(f'{prompt}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(output_data)

        print(f'completed {prompt}')

