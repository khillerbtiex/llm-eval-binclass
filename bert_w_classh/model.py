import pandas as pd

# Example DataFrame
data = {
    "text_seq": [
        "The model gave the right ingredients.",
        "The response was incomplete.",
    ],
    "helpfulness": [4, 2],
    "honesty": [5, 3],
    "instruction_following": [4, 2],
    "truthfulness": [4, 3],
    "overall_score": [4, 2],
    "label": [1, 0],  # Binary labels (0 or 1)
}

df = pd.read_csv("output_sample.csv")
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(text):
    return tokenizer(
        text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )


# Tokenize the text sequences
tokenized_texts = df["text_seq"].values
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, texts, scores, labels):
        self.texts = [tokenize_function(txt) for txt in texts]
        self.scores = scores
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.texts[idx]
        # Tokenize the text sequence

        # Concatenate scores as additional features
        scores_tensor = torch.tensor(self.scores[idx], dtype=torch.float)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        return {
            "input_ids": item["input_ids"].squeeze(),
            "attention_mask": item["attention_mask"].squeeze(),
            "scores": scores_tensor,
            "labels": label_tensor,
        }


# Prepare scores and labels
scores = df[
    [
        "helpfulness",
        "honesty",
        "instruction_following",
        "truthfulness",
        "overall_score",
    ]
].values
labels = df["label"].values

# Create dataset
dataset = CustomDataset(tokenized_texts, scores, labels)
from transformers import BertModel


class BertClassifier(torch.nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = torch.nn.Linear(
            self.bert.config.hidden_size + 5, 256
        )  # +5 for the scores
        self.fc2 = torch.nn.Linear(256, 1)  # Binary output
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, scores):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined_input = torch.cat((pooled_output, scores), dim=1)
        x = self.fc1(combined_input)
        x = torch.nn.ReLU()(x)
        logits = self.fc2(x)
        return self.sigmoid(logits)


def freeze_bert_layers(model, unfreeze_last_n_layers=6):
    # Freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unfreeze the last unfreeze_last_n_layers transformer layers
    for layer in model.bert.encoder.layer[-unfreeze_last_n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


# Instantiate and freeze layers
model = BertClassifier()
freeze_bert_layers(model)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists("model.pt"):
    checkpoint = torch.load("model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

model = model.to(device)
from torch.utils.data import DataLoader
from tqdm import tqdm

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in tqdm(range(4)):  # Number of epochs
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = batch["scores"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, scores)
        loss = criterion(outputs.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    },
    "model.pt",
)
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np

# Ensure your model is in evaluation mode
model.eval()

# Lists to store predictions and labels
preds = []
labels = []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = batch["scores"].to(device)
        labels_ = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask, scores)

        # Append predictions and labels
        preds.append(outputs.cpu().detach().numpy())
        labels.append(labels_.cpu().detach().numpy())

# Concatenate results into arrays
preds = np.concatenate(preds)
labels = np.concatenate(labels)

# Apply a threshold to convert probabilities to binary predictions
threshold = 0.5
binary_preds = (preds > threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(labels, binary_preds)
precision = precision_score(labels, binary_preds)
f1 = f1_score(labels, binary_preds)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
