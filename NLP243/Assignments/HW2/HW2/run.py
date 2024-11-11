import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from seqeval.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

# used GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# loda
train_df = pd.read_csv('hw2_train.csv')

# init BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
bert_model = BertModel.from_pretrained("bert-large-uncased").to(device)

# max len == 32
MAX_LENGTH = 32

# Adject BERT encode texts
def encode_texts(texts, tokenizer, bert_model, device, max_length=MAX_LENGTH):
    inputs = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state  # Shape: (batch_size, max_length, hidden_dim)
    return embeddings, inputs["attention_mask"].sum(dim=1)

# Extracting BERT embeddings for sentences
utterances = train_df['utterances'].tolist()
embeddings, sequence_lengths = encode_texts(utterances, tokenizer, bert_model, device)

# Build a label mapping dictionary
unique_labels = set(label for tags in train_df['IOB Slot tags'] for label in tags.split())
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_index.items()}

# max_length length of BERT output
labels = train_df['IOB Slot tags'].apply(lambda x: [label_to_index[label] for label in x.split()])
labels_padded = nn.utils.rnn.pad_sequence(
    [torch.tensor(label + [label_to_index["O"]] * (MAX_LENGTH - len(label))) for label in labels], 
    batch_first=True
).to(device)

# split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_padded, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# def Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()

# def GRU 
class SlotTaggingModelGRU(nn.Module):
    def __init__(self, bert_hidden_dim, hidden_dim=131, output_dim=None, dropout_prob=0.23450624849590243, num_layers=2):
        super(SlotTaggingModelGRU, self).__init__()
        self.gru = nn.GRU(
            bert_hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.dropout(gru_out)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        gru_out = gru_out * attn_weights
        output = self.fc(gru_out)
        return output

# init
bert_hidden_dim = embeddings.shape[2]
output_dim = len(label_to_index)

model = SlotTaggingModelGRU(
    bert_hidden_dim=bert_hidden_dim,
    hidden_dim=131,
    output_dim=output_dim,
    dropout_prob=0.23450624849590243,
    num_layers=2
).to(device)

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = AdamW(model.parameters(), lr=0.0009950695432002095, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# train
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x).view(-1, output_dim)
        batch_y = batch_y.view(-1)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# eval used seqeval
def evaluate_model(model, test_loader, criterion, device, idx_to_label):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).view(-1, output_dim)
            batch_y = batch_y.view(-1)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = batch_y.cpu().numpy()

            preds = preds.reshape(batch_x.size(0), -1)
            labels = labels.reshape(batch_x.size(0), -1)

            for pred, label in zip(preds, labels):
                pred_tags = [idx_to_label[idx] for idx in pred if idx in idx_to_label]
                true_tags = [idx_to_label[idx] for idx in label if idx in idx_to_label]
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    f1 = f1_score(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))

    return total_loss / len(test_loader), f1

# train
num_epochs = 50  
train_losses, f1_scores = [], []

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, f1 = evaluate_model(model, test_loader, criterion, device, idx_to_label)
    scheduler.step()
    
    train_losses.append(train_loss)
    f1_scores.append(f1)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test F1 Score: {f1:.4f}")

# plt Loss & F1 Score 
plt.figure(figsize=(12, 5))
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(range(num_epochs), f1_scores, label="F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.title("F1 Score over Epochs")
plt.legend()
plt.show()

# get the output
test_df = pd.read_csv('hw2_test.csv')

def generate_submission_file(model, test_df, tokenizer, bert_model, idx_to_label, device, output_file="submission.csv"):
    model.eval()
    predictions = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            utterance = row["utterances"]
            inputs = tokenizer(utterance, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
            embeddings = bert_model(**inputs).last_hidden_state
            
            outputs = model(embeddings)
            
            if outputs.dim() == 2:
                outputs = outputs.unsqueeze(0)

            pred_labels = torch.argmax(outputs, dim=2).squeeze().cpu().numpy()
            pred_labels = [idx_to_label[label] for label in pred_labels[:len(inputs['input_ids'][0])]]

            tokens = tokenizer.tokenize(utterance)
            final_labels = []
            token_idx = 0

            for label in pred_labels:
                if token_idx >= len(tokens):
                    break
                if tokens[token_idx].startswith("##"):
                    token_idx += 1
                    continue
                final_labels.append(label if label != "O" else "O")
                token_idx += 1

            predictions.append(final_labels)

    submission_df = pd.DataFrame({"ID": test_df["ID"], "IOB Slot tags": [" ".join(tags) for tags in predictions]})
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file generated：{output_file}")

generate_submission_file(model, test_df, tokenizer, bert_model, idx_to_label, device)
