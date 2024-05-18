from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from transformers import AutoTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import random
from scipy.special import softmax
import torch
import numpy as np
from transformers import create_optimizer
import evaluate
from transformers import pipeline
import csv

def preprocess_function(examples):
    tokenizer_dict = tokenizer(examples["text"], truncation=True, max_length=128, add_special_tokens = True)
    tokenizer_dict['label'] = examples['label']
    return tokenizer_dict

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_data():
    tsv_file_path = 'politeness.tsv'

    train_x, train_y = [], []
    test_x, test_y = [], []
    val_x, val_y = [], []

    # Open the TSV file and read its contents, limited to 100 rows
    with open(tsv_file_path, newline='') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        # count = 0
        for row in reader:
            # Check the value of the 'split' column and process accordingly
            if row['split'] == 'train':
                # Check the value of the 'style' column and assign labels
                if row['style'] in ['P_0', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5']:
                    train_x.append(row['txt'])
                    train_y.append(0)
                elif row['style'] in ['P_6', 'P_7', 'P_8', 'P_9']:
                    train_x.append(row['txt'])
                    train_y.append(1)
            elif row['split'] == 'test':
                test_x.append(row['txt'])
                # Assuming 'test' data doesn't need labeling
            elif row['split'] == 'val':
                val_x.append(row['txt'])
                # Assuming 'val' data doesn't need labeling

    # Print the lengths of the datasets for verification

    x = train_x[:100000]
    y = train_y[:100000]

    data = []
    for i in range(len(x)):
        data.append({"text": x[i], "label": y[i]})

    # print(data)

    return x, y, data

x, y, data = preprocess_data()
print(data[:20])

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

train_data, test_data = train_test_split(data, test_size=0.1)
data = {"train": [preprocess_function(d) for d in train_data], "test": [preprocess_function(d) for d in test_data]}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {0: "IMPOLITE", 1: "POLITE"}
label2id = {"IMPOLITE": 0, "POLITE": 1}

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)
device = torch.device("cuda")
model.cuda()

training_args = TrainingArguments(
    output_dir="politeness_outputs_bert_base_uncased_100k",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()

text = "a dog catching a tennis ball at sunset in a yard , playing fetch ."
val_input = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**val_input).logits
    probabilities = softmax(logits.cpu(), axis=1)
    predicted_label = id2label[np.argmax(probabilities)]
    print(f"Predicted Label: {predicted_label}")
    print(f"Probabilities: {probabilities}")
    # print(probabilities)
