from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd

# Load the preprocessed dataset from the CSV file
df = pd.read_csv("preprocessed-tagalog-hatespeech-train-dataset-RoBERTa.csv")

# Filter out invalid inputs
df = df.dropna(subset=["text"])
df = df[df["text"].apply(lambda x: isinstance(x, str))]

# Encode labels
labels = df["label"].tolist()

# Map labels to integers
labels = [int(label) for label in labels]

# Tokenize the text data using the Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained("jcblaise/roberta-tagalog-large")
encoded_data = tokenizer.batch_encode_plus(
    df["text"].tolist(),
    truncation=True,
    max_length=256,
    padding=True,
    return_tensors="pt",
)

# Convert labels to tensor
labels = torch.tensor(labels)

# Create a PyTorch DataLoader for training the model
input_ids = encoded_data["input_ids"]
attention_masks = encoded_data["attention_mask"]

# Pad the sequences
input_ids_padded = pad_sequence(
    [seq.clone().detach() for seq in input_ids], batch_first=True
)
attention_masks_padded = pad_sequence(
    [seq.clone().detach() for seq in attention_masks], batch_first=True
)


dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 8
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

# Retrieve the number of labels
num_labels = torch.unique(labels).size(0)

# Fine-tune the pre-trained model on the hate speech detection task
model = RobertaForSequenceClassification.from_pretrained(
    "jcblaise/roberta-tagalog-large", num_labels=num_labels
)
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Set up mixed-precision training
scaler = torch.cuda.amp.GradScaler()

num_epochs = 30

# Set up learning rate scheduler
total_steps = len(dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for step, batch in enumerate(dataloader):
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        # Clear previously calculated gradients
        model.zero_grad()

        # Forward pass
        outputs = model(
            batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Update learning rate scheduler
        scheduler.step()

        running_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(
                f"Epoch: {epoch+1}, Step: {step+1}/{len(dataloader)}, Loss: {running_loss / 10}"
            )

    # Save the model after each epoch
    output_dir = f"./trained_model_epoch_{epoch+1}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved after Epoch {epoch+1}")

print("Training complete!")
