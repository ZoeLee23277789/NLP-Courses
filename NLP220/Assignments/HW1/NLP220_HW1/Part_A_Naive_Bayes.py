import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

# read CSV
file_path = "Books_rating.csv"
df = pd.read_csv(file_path)
# Delete score 3
df_filtered = df[df['review/score'].isin([1, 2, 4, 5])]

#  4.5 == pos 1.2 == neg
df_filtered['label'] = df_filtered['review/score'].apply(lambda x: 1 if x >= 4 else 0)

print(df_filtered[['review/score', 'label']].head())
#===========================================================================
label_counts = df_filtered['label'].value_counts()

plt.figure(figsize=(6,4))
plt.bar(label_counts.index, label_counts.values, color=['blue', 'orange'])
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Distribution of Classes/Labels in the Dataset')

# show result
plt.show()
print(df_filtered['label'].value_counts())
# ==========================================================================
#split
train_data, test_data = train_test_split(df_filtered, test_size=0.15, random_state=42, stratify=df_filtered['label'])
# check split
print("訓練集大小:", train_data.shape)
print("測試集大小:", test_data.shape)
# ============================================================================
print("Calculate the 1 feature  CountVectorizer")
from sklearn.feature_extraction.text import CountVectorizer
# remove NaN 
train_data = train_data.dropna(subset=['review/text'])
test_data = test_data.dropna(subset=['review/text'])
vectorizer_count = CountVectorizer(stop_words='english', max_features=3000)
X_train_count = vectorizer_count.fit_transform(train_data['review/text'])
X_test_count = vectorizer_count.transform(test_data['review/text'])
# ============================================================================
print("Calculate the 2 feature  TF-IDF")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer_tfidf.fit_transform(train_data['review/text'])
X_test_tfidf = vectorizer_tfidf.transform(test_data['review/text'])
# =============================================================================
print("Calculate the 3 feature Combine  [review/summary] & [review/text] using TfidfVectorizer")
train_data['combined_text'] = train_data['review/summary'] + " " + train_data['review/text']
test_data['combined_text'] = test_data['review/summary'] + " " + test_data['review/text']
# remove NaN 
train_data['combined_text'] = train_data['combined_text'].fillna('')
test_data['combined_text'] = test_data['combined_text'].fillna('')

# used TfidfVectorizer 
vectorizer_combined = TfidfVectorizer(stop_words='english', max_features=100)
X_train_combined = vectorizer_combined.fit_transform(train_data['combined_text'])
X_test_combined = vectorizer_combined.transform(test_data['combined_text'])
print("=======================================================================================")
# =============================================================================
# Run Naive Bayes
print("Run Naive Bayes Model")
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

# build Naive Bayes 
nb_model_count = MultinomialNB()
nb_model_tfidf = MultinomialNB()
nb_model_combined = MultinomialNB()

# # cal and print result
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # eval
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, f1, cm, train_time, inference_time

# tag data
y_train = train_data['label']
y_test = test_data['label']

# used Count Vectorizer feature
acc_count, f1_count, cm_count, train_time_count, inference_time_count = train_and_evaluate(nb_model_count, X_train_count, X_test_count, y_train, y_test)

# used Tfidf feature
acc_tfidf, f1_tfidf, cm_tfidf, train_time_tfidf, inference_time_tfidf = train_and_evaluate(nb_model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test)

# used Combined feature
acc_combined, f1_combined, cm_combined, train_time_combined, inference_time_combined = train_and_evaluate(nb_model_combined, X_train_combined, X_test_combined, y_train, y_test)
# print result
print("Count Vectorizer - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_count, f1_count, train_time_count, inference_time_count))
print("Tfidf Vectorizer - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_tfidf, f1_tfidf, train_time_tfidf, inference_time_tfidf))
print("Combined Features - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_combined, f1_combined, train_time_combined, inference_time_combined))

print("Confusion Matrix for Count Vectorizer:\n", cm_count)
print("Confusion Matrix for Tfidf Vectorizer:\n", cm_tfidf)
print("Confusion Matrix for Combined Features:\n", cm_combined)
print("End the Naive Bayes model")
print("=======================================================================================")
# ===========================================================================================================================
print("Run Decision Tree Model")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

#  build Decision Tree model
dt_model_count = DecisionTreeClassifier(max_depth=10,min_samples_leaf=5)
dt_model_tfidf = DecisionTreeClassifier(max_depth=10,min_samples_leaf=5)
dt_model_combined = DecisionTreeClassifier(max_depth=10,min_samples_leaf=5)

# cal and print result
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # eval
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, f1, cm, train_time, inference_time

# tag data
y_train = train_data['label']
y_test = test_data['label']

# # used Count Vectorizer feature
acc_count, f1_count, cm_count, train_time_count, inference_time_count = train_and_evaluate(dt_model_count, X_train_count, X_test_count, y_train, y_test)

# used Tfidf feature
acc_tfidf, f1_tfidf, cm_tfidf, train_time_tfidf, inference_time_tfidf = train_and_evaluate(dt_model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test)

# used Combined feature
acc_combined, f1_combined, cm_combined, train_time_combined, inference_time_combined = train_and_evaluate(dt_model_combined, X_train_combined, X_test_combined, y_train, y_test)

# print result
print("Count Vectorizer - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_count, f1_count, train_time_count, inference_time_count))
print("Tfidf Vectorizer - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_tfidf, f1_tfidf, train_time_tfidf, inference_time_tfidf))
print("Combined Features - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_combined, f1_combined, train_time_combined, inference_time_combined))

print("Confusion Matrix for Count Vectorizer:\n", cm_count)
print("Confusion Matrix for Tfidf Vectorizer:\n", cm_tfidf)
print("Confusion Matrix for Combined Features:\n", cm_combined)
print("End the Decision Tree model")
print("=========================================================================")
# ===========================================================================================================================
print("Run LinearSVC Model")
from joblib import parallel_backend
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

# build LinearSVC model
svc_model_count = LinearSVC()
svc_model_tfidf = LinearSVC()
svc_model_combined = LinearSVC()

# Training and prediction functions
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # eval
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, f1, cm, train_time, inference_time

# tag data
y_train = train_data['label']
y_test = test_data['label']

# Using joblib for multi-core parallel computing
with parallel_backend('threading', n_jobs=-1):  # used all the CPU
    # used Count Vectorizer feature
    acc_count, f1_count, cm_count, train_time_count, inference_time_count = train_and_evaluate(svc_model_count, X_train_count, X_test_count, y_train, y_test)

    # used Tfidf feature
    acc_tfidf, f1_tfidf, cm_tfidf, train_time_tfidf, inference_time_tfidf = train_and_evaluate(svc_model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test)

    # used Combined feature
    acc_combined, f1_combined, cm_combined, train_time_combined, inference_time_combined = train_and_evaluate(svc_model_combined, X_train_combined, X_test_combined, y_train, y_test)

# print result
print("Count Vectorizer - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_count, f1_count, train_time_count, inference_time_count))
print("Tfidf Vectorizer - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_tfidf, f1_tfidf, train_time_tfidf, inference_time_tfidf))
print("Combined Features - Accuracy: {:.4f}, F1: {:.4f}, Training time: {:.4f}s, Inference time: {:.4f}s".format(acc_combined, f1_combined, train_time_combined, inference_time_combined))

print("Confusion Matrix for Count Vectorizer:\n", cm_count)
print("Confusion Matrix for Tfidf Vectorizer:\n", cm_tfidf)
print("Confusion Matrix for Combined Features:\n", cm_combined)
print("End the LinearSVC model")
print("=========================================================================")