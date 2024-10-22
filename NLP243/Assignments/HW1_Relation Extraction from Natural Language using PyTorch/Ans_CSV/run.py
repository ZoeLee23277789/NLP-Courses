import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import sys

# 檢查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的設備:", device)

# 從命令列引數讀取檔案路徑
train_file_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

# Load the file
train_df = pd.read_csv(train_file_path)

# Fill up the Nan to ""
train_df['CORE RELATIONS'] = train_df['CORE RELATIONS'].fillna('')

# strings into lists
train_df['CORE RELATIONS'] = train_df['CORE RELATIONS'].apply(lambda x: x.split() if isinstance(x, str) else [])
print(train_df.head())


# Init the BOW，max_features == 1000
vectorizer = CountVectorizer(max_features=1000)
print(vectorizer)
# UTTERANCES to array
X_train = vectorizer.fit_transform(train_df['UTTERANCES']).toarray()
print(X_train)
# print the shape
# [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
print(X_train.shape)  #  (sample, features)(2312, 1000)

mlb = MultiLabelBinarizer() # have mult-label

y_train = mlb.fit_transform(train_df['CORE RELATIONS']) # label to num arr

print(y_train.shape)  # output = (sample , total label)
print(mlb.classes_)    # label class name


# split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# the size of the dataset
print(X_train.shape, X_val.shape)


# Data -> PyTorch vector -> move to GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# MLP model
class MLPRelationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRelationModel, self).__init__()
        self.Layer1 = nn.Linear(input_dim, 256)  # hidden_layer 256
        self.Layer2 = nn.Linear(256, 128)        # hidden_layer 128
        self.Layer3 = nn.Linear(128, output_dim) # output_layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.Layer1(x))
        x = self.dropout(x)
        x = self.relu(self.Layer2(x))
        x = self.Layer3(x)
        return x

input_dim = X_train.shape[1]
output_dim = len(mlb.classes_)

# init model
model = MLPRelationModel(input_dim, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()  # Lossfunction - mult
optimizer = optim.Adam(model.parameters(), lr=0.0027414732532805475)  # Learning


# train
for epoch in range(100):  
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')


# read the csv file 
test_df = pd.read_csv(test_file_path)
X_test = vectorizer.transform(test_df['UTTERANCES']).toarray()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# used model to eval
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (torch.sigmoid(outputs) > 0.5).float().cpu()

# predict
predicted_relations = mlb.inverse_transform(predicted.numpy())

# create the final ans
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'CORE RELATIONS': ['none' if len(relations) == 0 else ' '.join(relations) for relations in predicted_relations]
})

# save the final ans
submission_df.to_csv('Final_submission.csv', index=False)
print("提交文件已成功生成，無法預測的行填充為 'none'！")