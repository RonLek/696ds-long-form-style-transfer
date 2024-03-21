from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import evaluate
from scipy.special import softmax
import torch
import numpy as np

def preprocess_function(examples):
    tokenizer_dict = tokenizer(examples["text"], truncation=True, max_length=512, add_special_tokens = True)
    tokenizer_dict['label'] = examples['label']
    return tokenizer_dict

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

data = []
with open('PWKP_108016') as f:
  raw_data = f.readlines()
  print(raw_data[-1])
  cur = []
  for line in raw_data:
    if line != '\n':
      cur.append(line)
    else:
      data.append({"text": cur[0][:-2], "label": 0})
      data.append({"text": cur[1][:-2], "label": 1})
      cur.clear()
print(data[-5:])
print(len(data))

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

train_data, test_data = train_test_split(data, test_size=0.1)
data = {"train": [preprocess_function(d) for d in train_data], "test": [preprocess_function(d) for d in test_data]}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {0: "COMPLEX", 1: "SIMPLE"}
label2id = {"COMPLEX": 0, "SIMPLE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-large-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

device = torch.device("cuda")
model.cuda()
training_args = TrainingArguments(
    output_dir="simplicity_outputs_bert_large_uncased",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
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

text = "In 1801 Quintana produced a tragedy, El Duque de Viseo, founded on M. G. Lewis's Castle Spectre; his Pelayo (1805), written on a patriotic theme, was more successful."
val_input = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**val_input).logits
    probabilities = softmax(logits.cpu(), axis=1)
    print(probabilities)

