import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Load the test data from CSV
data = pd.read_csv("preprocessed-tagalog-hatespeech-tweets-final-RoBERTa.csv")

# Check if column 1 is empty or contains non-string values and replace with a default value
X = data.iloc[:, 0].fillna("default_value").astype(str)

# Split the data into features and labels
y_true = data.iloc[:, 1]  # Assuming the true labels are in the second column

# Load pre-trained RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("trained_model_epoch_30")
model = RobertaForSequenceClassification.from_pretrained("trained_model_epoch_30")

# Tokenize input text
X_tokens = tokenizer.batch_encode_plus(
    X.tolist(), padding=True, truncation=True, return_tensors="pt"
)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert input tokens to tensors
input_ids = X_tokens["input_ids"]
attention_mask = X_tokens["attention_mask"]
labels = torch.tensor(y_true)

# Create a PyTorch DataLoader for testing the model
batch_size = 4  # Specify the desired batch size
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

# Pass tokenized input through the model for classification
with torch.no_grad():
    predictions = []
    for batch in dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)

        logits = model(batch_input_ids, attention_mask=batch_attention_mask)[0]
        batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(batch_predictions)

# Get predicted labels
y_pred = predictions

# Add labels and predictions to the DataFrame
data["True_Labels"] = y_true
data["Model_Predictions"] = y_pred
data["Match"] = y_true == y_pred

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the DataFrame to a CSV file
data.to_csv("predictions.csv", index=False)
