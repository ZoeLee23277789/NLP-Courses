print("Q1")
import nltk
from nltk.corpus import twitter_samples
import re
from collections import Counter
import csv

# Download required resources if not already present
nltk.download('twitter_samples')

# Load Twitter Corpus
tweets = twitter_samples.strings('tweets.20150430-223406.json')

# Data Cleaning: Remove Tweet handles, URLs, hashtags, and short sentences
cleaned_tweets = []
url_pattern = re.compile(r'https?://\S+|www\.\S+')
handle_pattern = re.compile(r'@\w+')
hashtag_pattern = re.compile(r'#\w+')

for tweet in tweets:
    # Remove URLs and handles
    tweet = url_pattern.sub('', tweet)
    tweet = handle_pattern.sub('', tweet)
    # Remove hashtags
    tweet = hashtag_pattern.sub('', tweet)
    # Tokenize and check token count
    tokens = tweet.split()
    if len(tokens) >= 4:
        cleaned_tweets.append(' '.join(tokens))

# Create a frequency distribution
all_words = ' '.join(cleaned_tweets).split()
freq_dist = Counter(all_words)

# Rank/ frequency profile
ranked_words = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
rank_freq_profile = [
    {"word": word, "rank": rank + 1, "freq": freq}
    for rank, (word, freq) in enumerate(ranked_words)
]

# Save the top-100 rank/frequency profile to a CSV file
output_file = 'rank_freq_profile.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Rank', 'Frequency'])
    for entry in rank_freq_profile[:100]:
        writer.writerow([entry["word"], entry["rank"], entry["freq"]])

# Compute the percentage of corpus size made up by the top-10 words
total_words = sum(freq_dist.values())
top_10_freq = sum([freq for _, freq in ranked_words[:10]])
percentage_top_10 = (top_10_freq / total_words) * 100

# Output results
print(f"Rank/Frequency profile saved to '{output_file}'.")
print(f"Percentage of corpus size made up by top-10 words: {percentage_top_10:.2f}%")
print("---------------------------------------------------------------------------------")
print("Q2")




import pandas as pd
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics.association import BigramAssocMeasures, TrigramAssocMeasures
from nltk.tokenize import word_tokenize

# 下載 NLTK 資源（如果尚未下載）
nltk.download('punkt')

# 加載數據
file_path = './semeval-2017-train.csv'
data = pd.read_csv(file_path, sep='\t', header=0, names=['label', 'text'])

# 分離正面和負面樣本
positive_samples = data[data['label'] > 0]['text'].tolist()
negative_samples = data[data['label'] < 0]['text'].tolist()

# Tokenize 文本樣本
positive_tokens = [word_tokenize(text) for text in positive_samples]
negative_tokens = [word_tokenize(text) for text in negative_samples]

# 展平 tokens 列表（將所有樣本的 token 合併為一個列表）
positive_tokens_flat = [token for sublist in positive_tokens for token in sublist]
negative_tokens_flat = [token for sublist in negative_tokens for token in sublist]

# 使用 NLTK 的 Collocation Finder 找出 bi-grams 和 tri-grams
bigram_measures = BigramAssocMeasures()
trigram_measures = TrigramAssocMeasures()

# 正面情緒 bi-grams 和 tri-grams
positive_bigram_finder = BigramCollocationFinder.from_words(positive_tokens_flat)
positive_trigram_finder = TrigramCollocationFinder.from_words(positive_tokens_flat)

positive_bigrams = positive_bigram_finder.nbest(bigram_measures.pmi, 10)
positive_trigrams = positive_trigram_finder.nbest(trigram_measures.pmi, 10)

# 負面情緒 bi-grams 和 tri-grams
negative_bigram_finder = BigramCollocationFinder.from_words(negative_tokens_flat)
negative_trigram_finder = TrigramCollocationFinder.from_words(negative_tokens_flat)

negative_bigrams = negative_bigram_finder.nbest(bigram_measures.pmi, 10)
negative_trigrams = negative_trigram_finder.nbest(trigram_measures.pmi, 10)

# 輸出結果
print("Top-10 Positive Sentiment Bigrams:", positive_bigrams)
print("Top-10 Positive Sentiment Trigrams:", positive_trigrams)
print("Top-10 Negative Sentiment Bigrams:", negative_bigrams)
print("Top-10 Negative Sentiment Trigrams:", negative_trigrams)




print("---------------------------------------------------------------------------------")
print("Q3")
from xml.dom import minidom
import csv
import json

# Load the XML file using DOM parsing
xml_file_path = './movies-new.xml'
dom_tree = minidom.parse(xml_file_path)

# Find all movies
movies = dom_tree.getElementsByTagName("movie")

# Extract required details and prepare data
movie_data = []
for movie in movies:
    title = movie.getAttribute("title")
    year = movie.getElementsByTagName("year")[0].firstChild.data
    rating = movie.getElementsByTagName("rating")[0].firstChild.data
    description = movie.getElementsByTagName("description")[0].firstChild.data.strip()
    movie_data.append({"title": title, "year": int(year), "rating": rating, "description": description})

# Save the data to a CSV file
csv_file_path = './movies-new.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["title", "year", "rating", "description"])
    writer.writeheader()
    writer.writerows(movie_data)

# Save the data to a JSON file
json_file_path = './movies-new.json'
with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
    json.dump(movie_data, jsonfile, indent=4)

print(f"Data saved to {csv_file_path} and {json_file_path}")
import xml.etree.ElementTree as ET

# Load the XML file using Element-Tree
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Find all Thriller movies and print their title and year
thriller_movies = []
for genre in root.findall(".//genre[@category='Thriller']"):
    for movie in genre.findall(".//movie"):
        title = movie.attrib.get("title")
        year = movie.find("year").text
        thriller_movies.append({"title": title, "year": year})

print("Thriller Movies:")
for thriller_movie in thriller_movies:
    print(f"Title: {thriller_movie['title']}, Year: {thriller_movie['year']}")
