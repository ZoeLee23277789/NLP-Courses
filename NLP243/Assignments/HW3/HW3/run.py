import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.optim import AdamW
from datasets import load_dataset
from collections import Counter
import numpy as np
import re

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加載 Penn Treebank 數據集
ptb = load_dataset('ptb-text-only/ptb_text_only', download_mode="force_redownload")

# Tokenize and Vocabulary Building
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

# 構建詞彙表
vocab = build_vocab(ptb['train'])
vocab_size = len(vocab)
pad_token_idx = vocab['<PAD>']

# 將文本轉換為索引序列
def encode_text(text, vocab):
    return [vocab[word] for word in tokenize(text) if word in vocab]

# 將數據集處理為索引序列
train_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb['train']]
val_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb['validation']]
test_data = [torch.tensor(encode_text(example['sentence'], vocab)) for example in ptb['test']]

# DataLoader Preparation
def collate_batch(batch):
    sequences = pad_sequence(batch, batch_first=True, padding_value=pad_token_idx)
    return sequences[:, :-1], sequences[:, 1:]  # Inputs and targets

# 定義 GRU 語言模型
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

# 模型參數配置
embedding_dim = 100
hidden_dim = 256
num_layers = 2
dropout_prob = 0.2
batch_size = 64
learning_rate = 0.001
epochs = 20

# 初始化模型
model = LanguageModelGRU(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    pad_idx=pad_token_idx,
    dropout_prob=dropout_prob
).to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 訓練數據加載器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# 訓練模型
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

# 驗證模型
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

# 訓練迴圈
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    val_perplexity = evaluate_model(model, val_loader, criterion)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")

# 測試模型
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_perplexity = evaluate_model(model, test_loader, criterion)
print(f"Test Perplexity: {test_perplexity:.4f}")
