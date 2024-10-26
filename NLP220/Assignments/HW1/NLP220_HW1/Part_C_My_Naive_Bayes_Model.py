import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# read CSV
file_path = "Books_rating.csv"
df = pd.read_csv(file_path)

# Delete score 3
df_filtered = df[df['review/score'].isin([1, 2, 4, 5])]

# 4.5 == pos 1.2 == neg
df_filtered['label'] = df_filtered['review/score'].apply(lambda x: 1 if x >= 4 else 0)

# remove NaN 
df_filtered = df_filtered.dropna(subset=['review/text'])

#split
train_data, test_data = train_test_split(df_filtered, test_size=0.15, random_state=42, stratify=df_filtered['label'])

class MyNaiveBayes:
    def __init__(self, alpha=10):
        self.alpha = alpha
        self.class_priors = None
        self.feature_log_prob = None
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        # P(y)
        self.class_priors = np.zeros(len(self.classes))
        for idx, c in enumerate(self.classes):
            self.class_priors[idx] = np.sum(y == c) / len(y)
        
        # used scipy 
        feature_count = np.zeros((len(self.classes), X.shape[1]))
        for idx, c in enumerate(self.classes):
            #matrix
            feature_count[idx, :] = X[y == c].sum(axis=0)

        # used alpha
        self.feature_log_prob = np.log((feature_count + self.alpha) / 
                                       (feature_count.sum(axis=1, keepdims=True) + self.alpha * X.shape[1]))
    
    def predict(self, X):
        log_probs = np.log(self.class_priors) + X @ self.feature_log_prob.T
        return self.classes[np.argmax(log_probs, axis=1)]

# cal and print result
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(f"{model_name} F1 Score: {f1}")
    print(f"{model_name} Confusion Matrix:\n{cm}\n")
# 1. CountVectorizer
print("Calculate the 1st feature: CountVectorizer")
vectorizer_count = CountVectorizer(stop_words='english', max_features=3000)
X_train_count = vectorizer_count.fit_transform(train_data['review/text'])
X_test_count = vectorizer_count.transform(test_data['review/text'])

# My Naive Bayes
print("My Naive Bayes ANS")
my_nb = MyNaiveBayes()
my_nb.fit(X_train_count, train_data['label'].values)  
my_preds = my_nb.predict(X_test_count) 
evaluate_model(test_data['label'], my_preds, "My Naive Bayes - CountVectorizer")

# Sklearn Naive Bayes
print("Sklearn ANS")
sklearn_nb = MultinomialNB()
sklearn_nb.fit(X_train_count, train_data['label'])
sklearn_preds = sklearn_nb.predict(X_test_count)
evaluate_model(test_data['label'], sklearn_preds, "Sklearn Naive Bayes - CountVectorizer")

# 2. TF-IDF
print("Calculate the 2nd feature: TF-IDF")
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer_tfidf.fit_transform(train_data['review/text'])
X_test_tfidf = vectorizer_tfidf.transform(test_data['review/text'])

# My Naive Bayes
print("My Naive Bayes ANS")
my_nb.fit(X_train_tfidf, train_data['label'].values)  
my_preds = my_nb.predict(X_test_tfidf)  
evaluate_model(test_data['label'], my_preds, "My Naive Bayes - TF-IDF")

# Sklearn Naive Bayes
print("Sklearn ANS")
sklearn_nb.fit(X_train_tfidf, train_data['label'])
sklearn_preds = sklearn_nb.predict(X_test_tfidf)
evaluate_model(test_data['label'], sklearn_preds, "Sklearn Naive Bayes - TF-IDF")

# 3. Combined [review/summary] & [review/text] using TfidfVectorizer
print("Calculate the 3rd feature: Combine [review/summary] & [review/text] using TfidfVectorizer")
train_data['combined_text'] = train_data['review/summary'].fillna('') + " " + train_data['review/text']
test_data['combined_text'] = test_data['review/summary'].fillna('') + " " + test_data['review/text']

vectorizer_combined = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_combined = vectorizer_combined.fit_transform(train_data['combined_text'])
X_test_combined = vectorizer_combined.transform(test_data['combined_text'])

# My Naive Bayes
print("My Naive Bayes ANS")
my_nb.fit(X_train_combined, train_data['label'].values)  
my_preds = my_nb.predict(X_test_combined)  
evaluate_model(test_data['label'], my_preds, "My Naive Bayes - Combined Text")

# Sklearn Naive Bayes
print("Sklearn ANS")
sklearn_nb.fit(X_train_combined, train_data['label'])
sklearn_preds = sklearn_nb.predict(X_test_combined)
evaluate_model(test_data['label'], sklearn_preds, "Sklearn Naive Bayes - Combined Text")

