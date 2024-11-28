import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
# read data
Data_path = './spam_sms_collection.tsv'
data = pd.read_csv(Data_path, sep='\t', header=None, names=['label', 'message'])
print(data)
# Data preprocessing spam = 1, ham = 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print(data.head(300))
# split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
print("X_train = " , X_train)
print("X_test = " , X_test)
# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9 ,max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# eval
# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)
y_pred_dt = dt_model.predict(X_test_tfidf)
dt_f1 = f1_score(y_test, y_pred_dt)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
nb_report = classification_report(y_test, y_pred_nb)
dt_report = classification_report(y_test, y_pred_dt)
# get the result
with open('./model_results.txt', 'w') as file:
    file.write("Naive Bayes Model:\n")
    file.write(f"Accuracy: {nb_accuracy}\n")
    file.write(f"F1 Score: {nb_f1}\n")
    file.write("Classification Report:\n")
    file.write(nb_report)
    file.write("\n\nDecision Tree Model:\n")
    file.write(f"Accuracy: {dt_accuracy}\n")
    file.write(f"F1 Score: {dt_f1}\n")
    file.write("Classification Report:\n")
    file.write(dt_report)
    
print("Results have been saved to 'model_results.txt'")