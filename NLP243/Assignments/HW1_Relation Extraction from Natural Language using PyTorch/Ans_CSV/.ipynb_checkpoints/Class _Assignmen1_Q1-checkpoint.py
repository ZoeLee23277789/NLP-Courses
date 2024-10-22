import csv
import pandas as pd
from collections import Counter
import nltk
import spacy
from nltk.corpus import twitter_samples
from nltk import pos_tag, word_tokenize
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, classification_report
nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")

csv_file_path = 'data-text.csv'
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    csv_data = [row for idx, row in enumerate(reader) if idx < 1000]
# 1
df = pd.read_csv(csv_file_path)
First_1000 = df.head(1000)
start = len(df) // 2 - 250
end = start + 500
middlesamples = df.iloc[start:end].copy()
middlesamples = middlesamples.drop(columns=['Low', 'High', 'Comments']).applymap(lambda x: str(x).lower())
middlesamplespath = 'middlesamples.tsv'
middlesamples.to_csv(middlesamplespath, sep='\t', index=False)

#2
tweets = twitter_samples.strings('tweets.20150430-223406.json')

nltk_adj = 0
nltk_n = 0

nltk_nouns_counter = Counter()
spacy_nouns_counter = Counter()

spacy_adj = 0
spacy_n = 0
for tweet in tweets:
    nltk_tokens = word_tokenize(tweet)
    nltk_tags = pos_tag(nltk_tokens)
    for word, tag in nltk_tags:
        if tag.startswith('JJ'):
            nltk_adj += 1
        elif tag.startswith('NN'):
            nltk_n += 1
            nltk_nouns_counter[word.lower()] += 1

    doc = nlp(tweet)
    for token in doc:
        if token.pos_ == 'ADJ':
            spacy_adj += 1
        elif token.pos_ == 'NOUN':
            spacy_n += 1
            spacy_nouns_counter[token.text.lower()] += 1

nltk_top_n = nltk_nouns_counter.most_common(10)
spacy_top_n = spacy_nouns_counter.most_common(10)

#3
iris = load_iris()

X = iris.data
y = iris.target

binary = y < 2

y_b = y[binary]
X_b = X[binary]


X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

# Save
output_text = f"""
Part 2 
- Total NLTK Adj: {nltk_adj}
- Total NLTK N: {nltk_n}
- Total Spacy Adj: {spacy_adj}
- Total Spacy N: {spacy_n}

- NLTK Top 10 N: {nltk_top_nouns}
- Spacy Top 10 N: {spacy_top_nouns}

Part 3 
- Accuracy: {accuracy}
- Macro F1 Score: {macro_f1}

"""

with open('assignment_output.txt', 'w') as file:
    file.write(output_text)
