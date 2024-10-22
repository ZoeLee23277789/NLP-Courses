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

# 讀取訓練資料
train_df = pd.read_csv(train_file_path)
train_df['CORE RELATIONS'] = train_df['CORE RELATIONS'].fillna('')
train_df['CORE RELATIONS'] = train_df['CORE RELATIONS'].apply(lambda x: x.split() if isinstance(x, str) else [])

# 初始化 BOW，設定 max_features 為 1000
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_df['UTTERANCES']).toarray()
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['CORE RELATIONS'])

# 分割訓練和驗證集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# 建立 PyTorch 資料集和資料載入器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定義模型
class MLPRelationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRelationModel, self).__init__()
        self.Layer1 = nn.Linear(input_dim, 256)
        self.Layer2 = nn.Linear(256, 128)
        self.Layer3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.Layer1(x))
        x = self.dropout(x)
        x = self.relu(self.Layer2(x))
        x = self.Layer3(x)
        return x

# 初始化模型、損失函數和優化器
input_dim = X_train.shape[1]
output_dim = len(mlb.classes_)
model = MLPRelationModel(input_dim, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0027414732532805475)

# 訓練模型
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
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

# 讀取測試資料並進行預測
test_df = pd.read_csv(test_file_path)
X_test = vectorizer.transform(test_df['UTTERANCES']).toarray()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (torch.sigmoid(outputs) > 0.5).float().cpu()
predicted_relations = mlb.inverse_transform(predicted.numpy())

# 生成並保存提交文件
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'CORE RELATIONS': ['none' if len(relations) == 0 else ' '.join(relations) for relations in predicted_relations]
})
submission_df.to_csv(output_file_path, index=False)
print("提交文件已成功生成！")
