

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import create_optimizer
from transformers import AdamW
import evaluate
import numpy as np
from transformers.keras_callbacks import KerasMetricCallback
import csv
from transformers import pipeline
import pickle

def preprocess_data():
    tsv_file_path = 'politeness.tsv'

    # Initialize empty lists for train, test, and validation data
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

    x = train_x
    y = train_y

    print("First 10 sentences and labels:")
    for i in range(10):
        print(f"Label: {y[i]}, Sentence: {x[i]}")
    print("\n")
    print('Train Data Length:', len(train_x), len(train_y))
    print('Test Data Length:', len(test_x))
    print('Validation Data Length:', len(val_x))
    print("\n")
    print("total lenght", len(train_x) + len(test_x) + len(val_x))
    print("% of train", len(train_x) / (len(train_x) + len(test_x) + len(val_x)))
    print("% of test", len(test_x) / (len(train_x) + len(test_x) + len(val_x)))
    print("% of val", len(val_x) / (len(train_x) + len(test_x) + len(val_x)))

    return x, y

def construct_encodings(x, tokenizer, truncation=True, padding=True):
    return tokenizer(x, return_tensors='tf', truncation=True, padding=True)

def construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))

def split_dataset(tfdataset, x):
    TEST_SPLIT = 0.2
    BATCH_SIZE = 8

    train_size = int(len(x) * (1 - TEST_SPLIT))

    tfdataset = tfdataset.shuffle(len(x))
    tfdataset_train = tfdataset.take(train_size)
    tfdataset_test = tfdataset.skip(train_size)

    tfdataset_train = tfdataset_train.batch(BATCH_SIZE)
    tfdataset_test = tfdataset_test.batch(BATCH_SIZE)

    return tfdataset_train, tfdataset_test

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def create_predictor(model, model_name):
  tkzr = AutoTokenizer.from_pretrained('bert-base-uncased')
  def predict_proba(text):
      x = [text]

      encodings = construct_encodings(x, tkzr)
      tfdataset = construct_tfdataset(encodings)
      tfdataset = tfdataset.batch(1)

      preds = model.predict(tfdataset).logits
      preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
      return preds[0][0], preds[0][1]

  return predict_proba

x, y = preprocess_data()

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

MAX_LEN = 512
sentence = x[0]
inputs = tokenizer(sentence, max_length=MAX_LEN, truncation=True, padding=True)

print(f'sentence: \'{sentence}\'')
print(f'input ids: {inputs["input_ids"]}')
print(f'attention mask: {inputs["attention_mask"]}')
print("\n")
print(sentence)
print("\n")
encodings = construct_encodings(x, tokenizer, truncation=True, padding=True)

id2label = {0: "IMPOLITE", 1: "POLITE"}
label2id = {"IMPOLITE": 0, "POLITE": 1}

train_size = int(len(x) * (0.8)) #split is 80-20
batch_size = 8
num_epochs = 2
batches_per_epoch = train_size // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model = TFBertForSequenceClassification.from_pretrained("google-bert/bert-large-uncased", num_labels=2,
                                                        id2label=id2label, label2id=label2id)
tfdataset = construct_tfdataset(encodings, y)

tfdataset_train, tfdataset_test = split_dataset(tfdataset, x)

accuracy = evaluate.load("accuracy")
model.compile(optimizer=optimizer, metrics=['accuracy'])

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tfdataset_test)
model.fit(x=tfdataset_train, validation_data=tfdataset_test, epochs=2, callbacks=metric_callback)

BATCH_SIZE = 8
benchmarks = model.evaluate(tfdataset_test, return_dict=True, batch_size=BATCH_SIZE)
print(benchmarks)

#SAVE MODEL
MODEL_NAME = 'bert-base-uncased'
model.save_pretrained('./model/clf')
with open('./model/info.pkl', 'wb') as f:
    pickle.dump((MODEL_NAME, MAX_LEN), f)


text = "Mitchell has largely kept out of the spotlight since suffering a brain aneurysm rupture in 2015,Variety reported. She has since relearned how to walk and regularly hosted parties with `Joni jams` at her central California home, the outlet said."
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
classifier(text)

MODEL_NAME = 'bert-base-uncased'
clf = create_predictor(model, MODEL_NAME)
print('IMPOLITE.   POLITE.')
print(clf(text))

inputs = tokenizer(text, return_tensors="tf")
logits = model(**inputs).logits
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
print(text, "====>", model.config.id2label[predicted_class_id])
