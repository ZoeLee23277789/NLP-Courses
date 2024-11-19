import json
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# 加載數據
with open('arxiv_data.json', 'r') as f:
    data = json.load(f)

# 構建 DataFrame
df = pd.DataFrame({
    'title': data['titles'],
    'abstract': data['summaries'],
    'labels': data['terms']
})

# 預處理文本
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # 去除非字母字符
    text = text.lower()  # 全部轉小寫
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['abstract'] = df['abstract'].apply(preprocess_text)

# 將多標籤轉為二值矩陣
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

# 分割訓練集、驗證集、測試集
train_texts, test_texts, y_train, y_test = train_test_split(df['abstract'], y, test_size=0.15, random_state=42)
train_texts, val_texts, y_train, y_val = train_test_split(train_texts, y_train, test_size=0.1765, random_state=42)

# 使用 TF-IDF 表示文本特徵
tfidf = TfidfVectorizer(max_features=500)  # 限制特徵數量以提高速度
X_train = tfidf.fit_transform(train_texts).toarray()
X_val = tfidf.transform(val_texts).toarray()
X_test = tfidf.transform(test_texts).toarray()

# 使用最佳超參數初始化 Hist Gradient Boosting 模型
best_params = {
    'max_iter': 291,
    'learning_rate': 0.0235,
    'max_leaf_nodes': 96,
    'min_samples_leaf': 9
}
hgb_model = HistGradientBoostingClassifier(
    max_iter=best_params['max_iter'],
    learning_rate=best_params['learning_rate'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
multi_target_model = MultiOutputClassifier(hgb_model, n_jobs=-1)

# 訓練時間測量
start_time = time.time()
multi_target_model.fit(X_train, y_train)
train_time = time.time() - start_time

# 驗證集推理和評估
start_time = time.time()
y_val_pred = multi_target_model.predict(X_val)
inference_time_val = time.time() - start_time

val_f1_score = f1_score(y_val, y_val_pred, average='micro')
val_report = classification_report(y_val, y_val_pred, zero_division=0)
print(f"Validation F1 Score: {val_f1_score:.4f}")
print("Validation Classification Report:\n", val_report)

# 測試集推理和評估
start_time = time.time()
y_test_pred = multi_target_model.predict(X_test)
inference_time_test = time.time() - start_time

test_f1_score = f1_score(y_test, y_test_pred, average='micro')
test_report = classification_report(y_test, y_test_pred, zero_division=0)
print(f"\nTest F1 Score: {test_f1_score:.4f}")
print("Test Classification Report:\n", test_report)

# 顯示訓練和推理時間
print(f"\nTraining time: {train_time:.4f} seconds")
print(f"Validation inference time: {inference_time_val:.4f} seconds")
print(f"Test inference time: {inference_time_test:.4f} seconds")

# results.txt
output_path = "Results.txt"
with open(output_path, "w") as f:
    f.write("Validation Classification Report:\n")
    f.write(val_report)
    f.write("\n\nTest Classification Report:\n")
    f.write(test_report)

print(f"Results saved to {output_path}")

# FInd the Best Params down belowe

# import optuna
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.metrics import f1_score
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Optuna setting
# def objective(trial):
#     max_iter = trial.suggest_int("max_iter", 50, 300)
#     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
#     max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 100)
#     min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    
#     # init model
#     model = HistGradientBoostingClassifier(
#         max_iter=max_iter,
#         learning_rate=learning_rate,
#         max_leaf_nodes=max_leaf_nodes,
#         min_samples_leaf=min_samples_leaf,
#         random_state=42
#     )
    
#     multi_target_model = MultiOutputClassifier(model, n_jobs=-1)
#     # train
#     multi_target_model.fit(X_train, y_train)
#     # eval
#     y_val_pred = multi_target_model.predict(X_val)
#     score = f1_score(y_val, y_val_pred, average='micro')  # 使用微平均計算F1分數
    
#     return score
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)
# print("Best hyperparameters:", study.best_params)
# print("Best micro F1 score:", study.best_value)

# # used best_params to train
# best_params = study.best_params
# best_model = HistGradientBoostingClassifier(
#     max_iter=best_params["max_iter"],
#     learning_rate=best_params["learning_rate"],
#     max_leaf_nodes=best_params["max_leaf_nodes"],
#     min_samples_leaf=best_params["min_samples_leaf"],
#     random_state=42
# )
# best_multi_target_model = MultiOutputClassifier(best_model, n_jobs=-1)
# best_multi_target_model.fit(X_train, y_train)

# # val and test score
# y_val_pred = best_multi_target_model.predict(X_val)
# y_test_pred = best_multi_target_model.predict(X_test)

# val_report = classification_report(y_val, y_val_pred, zero_division=0)
# test_report = classification_report(y_test, y_test_pred, zero_division=0)

# # Result
# output_path = "Optimized_HistGradientBoosting_Results.txt"
# with open(output_path, "w") as f:
#     f.write("Validation Classification Report:\n")
#     f.write(val_report)
#     f.write("\n\nTest Classification Report:\n")
#     f.write(test_report)

# print(f"Results saved to {output_path}")
