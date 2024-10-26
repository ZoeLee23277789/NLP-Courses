import os
import pandas as pd
import optuna
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import os
import pandas as pd

# relative path
base_dir = os.path.join(".", "aclImdb_v1", "aclImdb")

def load_reviews(directory, label):
    reviews = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            review_text = file.read()
            # The file name format is [id]_[rating].txt, for example 200_8.txt
            id, rating = file_name.split('_')
            rating = rating.split('.')[0]  
            reviews.append([id, rating, review_text, label])
    return reviews

# Read the positive and negative comments from the training set and use os.path.join to dynamically build the path
train_pos_reviews = load_reviews(os.path.join(base_dir, "train", "pos"), 1)  # label=1 pos
train_neg_reviews = load_reviews(os.path.join(base_dir, "train", "neg"), 0)  # label =0 neg
# comb dataset
train_reviews = train_pos_reviews + train_neg_reviews

train_df = pd.DataFrame(train_reviews, columns=['id', 'rating', 'review_text', 'label'])
# save
train_df.to_csv(os.path.join(base_dir, "train", "train_reviews.csv"), index=False)
print("Get the train_reviews.csv")

test_pos_reviews = load_reviews(os.path.join(base_dir, "test", "pos"), 1)
test_neg_reviews = load_reviews(os.path.join(base_dir, "test", "neg"), 0)

test_reviews = test_pos_reviews + test_neg_reviews
test_df = pd.DataFrame(test_reviews, columns=['id', 'rating', 'review_text', 'label'])
test_df.to_csv(os.path.join(base_dir, "test", "test_reviews.csv"), index=False)
print("Get the test_reviews.csv")

train_df = pd.read_csv(os.path.join(base_dir, "train", "train_reviews.csv"))
test_df = pd.read_csv(os.path.join(base_dir, "test", "test_reviews.csv"))

#  train_test_split 
X_train, X_val, y_train, y_val = train_test_split(train_df['review_text'], train_df['label'], test_size=0.1, random_state=42)

#used TfidfVectorizer get n-gram feature
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')  # Example: bigrams, remove stopwords
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test_df['review_text'])

print("==================================================================================")

# Get the best hyperparameters form Optuna
best_params = {
    "Naive Bayes": {'alpha': 0.05450760213833489},
    "Logistic Regression": {'C': 89.67647856745852, 'solver': 'saga', 'max_iter': 1000},
    "Decision Tree": {'max_depth': 20, 'min_samples_split': 3},
    "Random Forest": {'n_estimators': 189, 'max_depth': 18, 'min_samples_split': 6},
    "KNN": {'n_neighbors': 7}
}

models = {
    "Naive Bayes": MultinomialNB(**best_params["Naive Bayes"]),
    "Logistic Regression": LogisticRegression(**best_params["Logistic Regression"]),
    "Decision Tree": DecisionTreeClassifier(**best_params["Decision Tree"]),
    "Random Forest": RandomForestClassifier(**best_params["Random Forest"]),
    "KNN": KNeighborsClassifier(**best_params["KNN"])
}
print("==================================================================================")
# save the best model
best_model_name = None
best_val_accuracy = 0

# train the best model
for model_name, model in models.items():
    print("model_name = " ,model_name)
    model.fit(X_train_tfidf, y_train)
    val_predictions = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"{model_name} Validation Accuracy: {val_accuracy}")
    
    # Save the best model on the validation set
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_name = model_name

# Evaluate on the test set using the best performing model on the validation set
print(f"\nBest Model: {best_model_name}")
best_model = models[best_model_name]
test_predictions = best_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(test_df['label'], test_predictions)
print(f"{best_model_name} Test Accuracy: {test_accuracy}")

print("==================================================================================")

# # Find the best hyperparameters using Optuna
# # 
# print("Define hyperparameters for each model to optimize")
# # 6. KNN
# def optimize_knn(trial):
#     n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
#     model = KNeighborsClassifier(n_neighbors=n_neighbors)
#     model.fit(X_train_tfidf, y_train)
#     val_predictions = model.predict(X_val_tfidf)
#     return accuracy_score(y_val, val_predictions)


# # 1. Naive Bayes
# def optimize_naive_bayes(trial):
#     alpha = trial.suggest_loguniform('alpha', 1e-3, 1e1)
#     model = MultinomialNB(alpha=alpha)
#     model.fit(X_train_tfidf, y_train)
#     val_predictions = model.predict(X_val_tfidf)
#     return accuracy_score(y_val, val_predictions)

# # 2. SVM
# def optimize_svm(trial):
#     C = trial.suggest_loguniform('C', 1e-4, 1e2)
#     kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
#     model = SVC(C=C, kernel=kernel)
#     model.fit(X_train_tfidf, y_train)
#     val_predictions = model.predict(X_val_tfidf)
#     return accuracy_score(y_val, val_predictions)

# # 3. Logistic Regression
# def optimize_logistic_regression(trial):
#     C = trial.suggest_loguniform('C', 1e-4, 1e2)
#     solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
#     model = LogisticRegression(C=C, solver=solver, max_iter=1000)
#     model.fit(X_train_tfidf, y_train)
#     val_predictions = model.predict(X_val_tfidf)
#     return accuracy_score(y_val, val_predictions)

# # 4. Decision Tree
# def optimize_decision_tree(trial):
#     max_depth = trial.suggest_int('max_depth', 2, 20)
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
#     model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
#     model.fit(X_train_tfidf, y_train)
#     val_predictions = model.predict(X_val_tfidf)
#     return accuracy_score(y_val, val_predictions)

# # 5. Random Forest
# def optimize_random_forest(trial):
#     n_estimators = trial.suggest_int('n_estimators', 10, 200)
#     max_depth = trial.suggest_int('max_depth', 2, 20)
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
#     model.fit(X_train_tfidf, y_train)
#     val_predictions = model.predict(X_val_tfidf)
#     return accuracy_score(y_val, val_predictions)

# print("Optimize the hyperparameters of each model")
# models = {
#     "Naive Bayes": optimize_naive_bayes,
#     "Logistic Regression": optimize_logistic_regression,
#     "Decision Tree": optimize_decision_tree,
#     "Random Forest": optimize_random_forest,
#     "KNN": optimize_knn,
# }

# best_models = {}
# for model_name, objective in models.items():
#     print(f"Optimizing {model_name}...")
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=50)
#     print(f"Best {model_name} parameters: {study.best_trial.params}")
#     print(f"Best validation accuracy: {study.best_trial.value}")
#     best_models[model_name] = study.best_trial.params

# print("Use the best hyperparameters for each model on the test set")
# for model_name, best_params in best_models.items():
#     print(f"\nTesting best {model_name} model on test set:")
    
#     if model_name == "Naive Bayes":
#         model = MultinomialNB(**best_params)
#     elif model_name == "Logistic Regression":
#         model = LogisticRegression(**best_params)
#     elif model_name == "Decision Tree":
#         model = DecisionTreeClassifier(**best_params)
#     elif model_name == "Random Forest":
#         model = RandomForestClassifier(**best_params)
#     elif model_name == "KNN":
#         model = KNN(**best_params)
#     model.fit(X_train_tfidf, y_train)
#     test_predictions = model.predict(X_test_tfidf)
#     test_accuracy = accuracy_score(test_df['label'], test_predictions)
#     print(f"{model_name} Test Accuracy: {test_accuracy}")

print("======================================= Finish all the Part B ===========================================")
