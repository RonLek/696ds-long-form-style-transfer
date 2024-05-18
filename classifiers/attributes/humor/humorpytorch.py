from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import random
from scipy.special import softmax
import torch
import numpy as np
from transformers import create_optimizer
import evaluate
from transformers import pipeline

def preprocess_function(examples):
    tokenizer_dict = tokenizer(examples["text"], truncation=True, max_length=128, add_special_tokens = True)
    tokenizer_dict['label'] = examples['label']
    return tokenizer_dict

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_data():
    def read_sentences_from_file(file_path):
        with open(file_path, 'r', errors='ignore') as file:
            sentences = file.readlines()
        return sentences

    def create_labeled_list(sentences, label):
        labeled_list = [(sentence.strip(), label) for sentence in sentences]
        return labeled_list

    file1 = "funny_train.txt"
    file2 = "romantic_train.txt"

    sentences_from_file1 = read_sentences_from_file(file1)
    sentences_from_file2 = read_sentences_from_file(file2)

    labeled_list = []
    while sentences_from_file1 or sentences_from_file2:
        if sentences_from_file1:
            sentence = random.choice(sentences_from_file1)
            labeled_list.append((sentence.strip(), 1))  # Label for file 1
            sentences_from_file1.remove(sentence)

        if sentences_from_file2:
            sentence = random.choice(sentences_from_file2)
            labeled_list.append((sentence.strip(), 0))  # Label for file 2
            sentences_from_file2.remove(sentence)

    random.shuffle(labeled_list)
    labels = [label for _, label in labeled_list]
    sentences = [sentence for sentence, _ in labeled_list]

    print('labels length',len(labels))
    print('sentences length',len(sentences))

    x = sentences

    y = labels


    print("List of labels:")
    print(y)
    print("\nList of corresponding sentences:")
    print(x)

    print('y length',len(y))
    print('x length',len(x))

    data = []
    for i in range(len(x)):
        data.append({"text": x[i], "label": y[i]})

    # print(data)

    return x, y, data

x, y, data = preprocess_data()
print(data[:20])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_data, test_data = train_test_split(data, test_size=0.1)
data = {"train": [preprocess_function(d) for d in train_data], "test": [preprocess_function(d) for d in test_data]}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {0: "NOTFUNNY", 1: "FUNNY"}
label2id = {"NOTFUNNY": 0, "FUNNY": 1}

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

device = torch.device("cuda")
model.cuda()

training_args = TrainingArguments(
    output_dir="humor_outputs_bert_base_uncased_1e-4_E6_B16",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
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
