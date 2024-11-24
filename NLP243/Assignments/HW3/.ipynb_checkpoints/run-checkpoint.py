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

def encode_text(text, vocab):
    return [vocab[word] for word in tokenize(text) if word in vocab]

def collate_batch(batch):
    sequences = pad_sequence(batch, batch_first=True, padding_value=pad_token_idx)
    return sequences[:, :-1], sequences[:, 1:]  # Inputs and targets

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

def evaluate_sentence_perplexities(model, dataloader, criterion):
    model.eval()
    sentence_perplexities = []
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            if inputs.size(1) == 0:
                perplexity = -1  # Placeholder for empty sequence
            else:
                logits = model(inputs)
                loss = criterion(logits[:, :-1, :].reshape(-1, vocab_size), inputs[:, 1:].reshape(-1))
                perplexity = np.exp(loss.item())
            sentence_perplexities.append((idx, perplexity))
    return sentence_perplexities

if __name__ == "__main__":
    # Command-line arguments
    output_csv = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"

    # Device and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ptb = load_dataset('ptb_text_only', split=['train', 'validation', 'test'], trust_remote_code=True)
    vocab = build_vocab(ptb[0])
    vocab_size = len(vocab)
    pad_token_idx = vocab['<PAD>']

    test_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb[2]]
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_batch)

    # Best model parameters
    best_params = {
        'hidden_dim': 451,
        'num_layers': 2,
        'learning_rate': 0.00030167213777739784,
        'batch_size': 64,
        'dropout_prob': 0.36114231298781496,
        'weight_decay': 0.009457391494207868
    }
    embedding_dim = 100
    model = LanguageModelGRU(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        pad_idx=pad_token_idx,
        dropout_prob=best_params['dropout_prob']
    ).to(device)

    # Load saved model weights if available
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)

    # Evaluate on test data
    test_perplexities = evaluate_sentence_perplexities(model, test_loader, criterion)

    # Save results to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "ppl"])
        for idx, perplexity in test_perplexities:
            writer.writerow([idx, perplexity])
    print(f"Submission file '{output_csv}' generated.")
