import csv
import re
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch import nn
from torch.optim import AdamW
import numpy as np
from datasets import load_dataset
import sys

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="is deprecated. Please use get_last_lr() to access the learning rate.")

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the dataset
ptb = load_dataset('ptb_text_only', split=['train', 'validation', 'test'], trust_remote_code=True)

# Tokenization and Vocabulary Building
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def build_vocab(dataset):
    counter = Counter()
    for example in dataset:
        tokens = tokenize(example['sentence'])
        counter.update(tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(counter.items())}
    vocab['<PAD>'] = len(vocab)
    return vocab

vocab = build_vocab(ptb[0])
vocab_size = len(vocab)
pad_token_idx = vocab['<PAD>']

# Convert text to sequences of indices
def encode_text(text, vocab):
    return [vocab[word] for word in tokenize(text) if word in vocab]

# Process each split
train_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb[0]]
val_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb[1]]
test_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb[2]]

# DataLoader preparation
def collate_batch(batch):
    sequences = pad_sequence(batch, batch_first=True, padding_value=pad_token_idx)
    return sequences[:, :-1], sequences[:, 1:]  # Inputs and targets

# Define the GRU Language Model with dropout
class LanguageModelGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx, dropout_prob):
        super(LanguageModelGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        logits = self.fc(gru_out)
        return logits

# Configuration 1 Hyperparameters
best_params = {
    'hidden_dim': 451,
    'num_layers': 2,
    'learning_rate': 0.00030167213777739784,
    'batch_size': 64,
    'dropout_prob': 0.36114231298781496,
    'weight_decay': 0.009457391494207868
}

# Model Initialization with Configuration 1
embedding_dim = 100
model = LanguageModelGRU(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    pad_idx=pad_token_idx,
    dropout_prob=best_params['dropout_prob']
).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
optimizer = AdamW(
    model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training and Evaluation Functions
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    return perplexity

# Training Loop with Early Stopping
train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_data, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_batch)
epochs = 100
best_val_perplexity = float("inf")
epochs_no_improve = 0
early_stop_patience = 5
print("Train epoch")
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    val_perplexity = evaluate_model(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
    
    # Update learning rate based on validation perplexity
    scheduler.step(val_perplexity)
    
    # Check for early stopping
    if val_perplexity < best_val_perplexity:
        best_val_perplexity = val_perplexity
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stop_patience:
            print("Early stopping triggered.")
            break

def evaluate_sentence_perplexities(model, dataloader, criterion, total_sentences):
    model.eval()
    sentence_perplexities = []
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            if inputs.size(1) == 0:  # Skip if sequence length is 0
                perplexity = -1  # Placeholder for empty sequence
            else:
                logits = model(inputs)
                # Adjust logits to match target sequence length
                loss = criterion(logits[:, :-1, :].reshape(-1, vocab_size), inputs[:, 1:].reshape(-1))
                perplexity = np.exp(loss.item())
            sentence_perplexities.append((idx, perplexity))
    return sentence_perplexities


# Prepare test loader and generate sentence-wise perplexities for submission
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_batch)
total_sentences = len(test_data)  # Number of sentences in the test set
test_perplexities = evaluate_sentence_perplexities(model, test_loader, criterion, total_sentences)

# Save perplexities to the output CSV file
if len(sys.argv) < 2:
    print("Usage: python run.py <output_csv>")
    sys.exit(1)

output_csv = sys.argv[1]
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "ppl"])  # Header as per requirement
    for idx, perplexity in test_perplexities:
        writer.writerow([idx, perplexity])

print(f"Submission file '{output_csv}' generated.")