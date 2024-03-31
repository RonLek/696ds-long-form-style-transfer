import pandas as pd
import os
import torch
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
from datasets import Dataset
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

domains = ['entertainment', 'finance', 'food', 'games', 'tech']
dataset = []
for index, d in enumerate(domains):
    with open(f'data/v2/non_parallel/{d}/data.000000000000.jsonl') as f:
        domain_data = f.readlines()
        for line in domain_data[:1000]:
            dataset.append({'text': json.loads(line)['text'], 'labels': index})

df_train = pd.DataFrame(dataset)
dataset = Dataset.from_pandas(df_train, split='train')
del df_train
print(dataset[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels = 5,  gradient_checkpointing=True, attention_window=512).to(device)
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=512)

def tokenize_function(batched_data):
    return tokenizer(batched_data['text'], padding='max_length', truncation=True, max_length=512)
    #result = tokenizer(batched_data['text'], padding='max_length', truncation=True, max_length=1024)
    #if tokenizer.is_fast:
    #    result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
    #return result

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'], batch_size=len(dataset))
#print(tokenized_datasets[:2])
#chunk_size = 1024
#def group_texts(batched_data):
#    concatenated_examples = {k: sum(batched_data[k], []) for k in batched_data.keys()}
#    total_length = len(concatenated_examples[list(batched_data.keys())[0]])
#    total_length = (total_length // chunk_size) * chunk_size
#    result = {k : [t[i: i+chunk_size] for i in range(0, total_length, chunk_size)] for k, t in concatenated_examples.items()}
#    result['labels'] = result['input_ids'].copy()
#    return result

#lm_datasets = tokenized_datasets.map(group_texts, batched=True)
#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

#dataset_split = lm_datasets.train_test_split(test_size=0.1)
dataset_split = tokenized_datasets.train_test_split(test_size=0.1)
print(dataset_split.shape)
print(dataset_split['train'][:2])
print(dataset_split['test'][:2])
from transformers import TrainingArguments

#batch_size = 4096
# Show the training loss with every epoch
#logging_steps = len(dataset_split["train"]) // batch_size
logging_steps = 1

training_args = TrainingArguments(
    output_dir="longformer-finetuned-non_parallel",
    num_train_epochs = 5,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=200,
    push_to_hub=False,
    fp16=True,
    logging_steps=4,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"]
)

trainer.train()

import math
eval_results = trainer.evaluate()

print(eval_results)
trainer.save_model()
