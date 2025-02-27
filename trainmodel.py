import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load your data
df = pd.read_csv('data.csv')

# Preprocess and tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    return text.lower()

df['cleaned_report'] = df['Report'].apply(preprocess_text)

# Encode target labels
df['CancerStatus'] = df['CancerStatus'].map({'no_cancer': 0, 'cancer': 1})

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_report'], df['CancerStatus'], test_size=0.2)

train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=512)
test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512)

# Define BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluation strategy
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size
    per_device_eval_batch_size=16,   # evaluation batch size
    num_train_epochs=3,              # number of epochs
    weight_decay=0.01,               # strength of weight decay
)

# Set up the Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_encodings,       # training dataset
    eval_dataset=test_encodings          # evaluation dataset
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('models/cervical_cancer_model')
tokenizer.save_pretrained('models/cervical_cancer_tokenizer')
