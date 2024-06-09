import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/emotion_dataset_raw.csv')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize data
def tokenize_data(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")


tokenized_data = data['Text'].apply(tokenize_data)

# Extract input_ids and attention_mask
data['input_ids'] = tokenized_data.apply(lambda x: x['input_ids'].squeeze(0))
data['attention_mask'] = tokenized_data.apply(lambda x: x['attention_mask'].squeeze(0))
data['labels'] = pd.Categorical(data['Emotion']).codes

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)


# Define dataset class
class NewsGroupDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Create train and test datasets
train_dataset = NewsGroupDataset({
    'input_ids': train_data['input_ids'].tolist(),
    'attention_mask': train_data['attention_mask'].tolist(),
}, train_data['labels'].tolist())

test_dataset = NewsGroupDataset({
    'input_ids': test_data['input_ids'].tolist(),
    'attention_mask': test_data['attention_mask'].tolist(),
}, test_data['labels'].tolist())

# Initialize the model
num_labels = len(pd.unique(data['Emotion']))
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# Define metrics computation
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    accuracy = accuracy_score(p.label_ids, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch"  # Note the change here from evaluation_strategy to eval_strategy
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train and evaluate the model
print("Starting training...")
trainer.train()
print("Training completed. Starting evaluation...")
results = trainer.evaluate()
print('Results:')
print(results)

# Extract accuracy from logs
train_acc = [entry['eval_accuracy'] for entry in trainer.state.log_history if 'eval_accuracy' in entry]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.show()
