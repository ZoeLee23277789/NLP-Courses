from collections import defaultdict
import math
import os

print(os.getcwd())  # 顯示目前的工作目錄

# 讀取數據
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return [line.strip().split() for line in data]

# 構建 <UNK> 標記，包含少於等於三次的詞
def process_unk_tokens(data, min_count=3):
    word_counts = defaultdict(int)
    for sentence in data:
        for word in sentence:
            word_counts[word] += 1

    processed_data = []
    for sentence in data:
        processed_sentence = [word if word_counts[word] >= min_count else "<UNK>" for word in sentence]
        processed_data.append(processed_sentence)
    return processed_data

# 構建 unigram 模型
def build_unigram_model(data):
    word_counts = defaultdict(int)
    total_count = 0
    for sentence in data:
        for word in sentence + ["<STOP>"]:  # 添加 <STOP> 到每句結尾
            word_counts[word] += 1
            total_count += 1
    # 計算機率
    unigram_prob = {word: count / total_count for word, count in word_counts.items()}
    return unigram_prob

# 計算困惑度
def calculate_perplexity(model, data):
    perplexity = 0
    total_words = 0
    for sentence in data:
        sentence = sentence + ["<STOP>"]  # 確保只包含 <STOP>，排除 <START>
        for word in sentence:
            prob = model.get(word, model.get("<UNK>", 1e-6))  # 使用 <UNK> 來處理未見詞
            perplexity += -math.log2(prob)
            total_words += 1
    return 2 ** (perplexity / total_words)

# 構建 bigram 模型
def build_bigram_model(data):
    bigram_counts = defaultdict(lambda: defaultdict(int))
    unigram_counts = defaultdict(int)
    for sentence in data:
        sentence = ["<START>"] + sentence + ["<STOP>"]
        for i in range(1, len(sentence)):
            unigram_counts[sentence[i - 1]] += 1
            bigram_counts[sentence[i - 1]][sentence[i]] += 1
        unigram_counts[sentence[-1]] += 1  # 更新最後詞的 unigram 計數
    # 計算條件機率
    bigram_prob = {w1: {w2: count / unigram_counts[w1] for w2, count in w2_dict.items()} 
                   for w1, w2_dict in bigram_counts.items()}
    return bigram_prob

def calculate_bigram_perplexity(model, data):
    perplexity = 0
    total_words = 0
    for sentence in data:
        sentence = ["<START>"] + sentence + ["<STOP>"]
        for i in range(1, len(sentence)):
            prob = model.get(sentence[i - 1], {}).get(sentence[i], 1e-6)
            perplexity += -math.log2(prob)
            total_words += 1
    return 2 ** (perplexity / total_words)

# 構建 trigram 模型
def build_trigram_model(data):
    trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bigram_counts = defaultdict(lambda: defaultdict(int))
    for sentence in data:
        sentence = ["<START>"] + sentence + ["<STOP>"]
        for i in range(2, len(sentence)):
            bigram_counts[sentence[i - 2]][sentence[i - 1]] += 1
            trigram_counts[sentence[i - 2]][sentence[i - 1]][sentence[i]] += 1
    # 計算條件機率
    trigram_prob = {w1: {w2: {w3: count / bigram_counts[w1][w2] for w3, count in w3_dict.items()} 
                         for w2, w3_dict in w2_dict.items()} 
                    for w1, w2_dict in trigram_counts.items()}
    return trigram_prob

def calculate_trigram_perplexity(model, data, bigram_model):
    perplexity = 0
    total_words = 0
    for sentence in data:
        sentence = ["<START>", "<START>"] + sentence + ["<STOP>"]
        for i in range(2, len(sentence)):
            # 對於第一個詞，使用 bigram 機率
            if i == 2:
                prob = bigram_model.get(sentence[i - 1], {}).get(sentence[i], 1e-6)
            else:
                prob = model.get(sentence[i - 2], {}).get(sentence[i - 1], {}).get(sentence[i], 1e-6)
            perplexity += -math.log2(prob)
            total_words += 1
    return 2 ** (perplexity / total_words)

# 測試函數
train_data = load_data('HW2/A2-Data/1b_benchmark.train.tokens')
dev_data = load_data('HW2/A2-Data/1b_benchmark.dev.tokens')
test_data = load_data('HW2/A2-Data/1b_benchmark.test.tokens') 

# 處理 <UNK> 標記
train_data = process_unk_tokens(train_data)
dev_data = process_unk_tokens(dev_data)

# 構建模型
unigram_model = build_unigram_model(train_data)
bigram_model = build_bigram_model(train_data)
trigram_model = build_trigram_model(train_data)

test_sentence = [["HDTV", "."]]
print("\nTesting with sentence: 'HDTV .'")
unigram_test_perplexity = calculate_perplexity(unigram_model, test_sentence)
bigram_test_perplexity = calculate_bigram_perplexity(bigram_model, test_sentence)
trigram_test_perplexity = calculate_trigram_perplexity(trigram_model, test_sentence, bigram_model)
print(f"Unigram Model - Test Perplexity (HDTV .): {unigram_test_perplexity}")
print(f"Bigram Model - Test Perplexity (HDTV .): {bigram_test_perplexity}")
print(f"Trigram Model - Test Perplexity (HDTV .): {trigram_test_perplexity}")
print("Final Ans=======================================\n")
# 計算並顯示訓練數據困惑度
train_perplexity = calculate_perplexity(unigram_model, train_data)
bigram_train_perplexity = calculate_bigram_perplexity(bigram_model, train_data)
trigram_train_perplexity = calculate_trigram_perplexity(trigram_model, train_data, bigram_model)

print(f"Unigram Model - Train Perplexity: {train_perplexity}")
print(f"Bigram Model - Train Perplexity: {bigram_train_perplexity}")
print(f"Trigram Model - Train Perplexity: {trigram_train_perplexity}")

# 計算並顯示開發數據困惑度
dev_perplexity = calculate_perplexity(unigram_model, dev_data)
bigram_dev_perplexity = calculate_bigram_perplexity(bigram_model, dev_data)
trigram_dev_perplexity = calculate_trigram_perplexity(trigram_model, dev_data, bigram_model)

print(f"Unigram Model - Development Perplexity: {dev_perplexity}")
print(f"Bigram Model - Development Perplexity: {bigram_dev_perplexity}")
print(f"Trigram Model - Development Perplexity: {trigram_dev_perplexity}")
import math
from collections import defaultdict

def calculate_interpolated_perplexity(unigram_model, bigram_model, trigram_model, data, lambda1, lambda2, lambda3):
    perplexity = 0
    total_words = 0
    for sentence in data:
        sentence = ["<START>", "<START>"] + sentence + ["<STOP>"]
        for i in range(2, len(sentence)):
            unigram_prob = unigram_model.get(sentence[i], unigram_model.get("<UNK>", 1e-6))
            bigram_prob = bigram_model.get(sentence[i - 1], {}).get(sentence[i], 1e-6)
            trigram_prob = trigram_model.get(sentence[i - 2], {}).get(sentence[i - 1], {}).get(sentence[i], 1e-6)
            
            # Interpolated probability with smoothing
            interpolated_prob = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob
            perplexity += -math.log2(interpolated_prob)
            total_words += 1
    return 2 ** (perplexity / total_words)

# Define five sets of lambda values
lambda_combinations = [
    (0.3, 0.3, 0.4),
    (0.1, 0.3, 0.6),
    (0.2, 0.4, 0.4),
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2)
]

# Variables to store the best results
best_dev_perplexity = float('inf')
best_lambda_combination = None

# Calculate perplexity for each set of lambda values
for lambda1, lambda2, lambda3 in lambda_combinations:
    train_perplexity = calculate_interpolated_perplexity(
        unigram_model, bigram_model, trigram_model, train_data, lambda1, lambda2, lambda3
    )
    dev_perplexity = calculate_interpolated_perplexity(
        unigram_model, bigram_model, trigram_model, dev_data, lambda1, lambda2, lambda3
    )
    
    print(f"Lambda values (λ1={lambda1}, λ2={lambda2}, λ3={lambda3}) - Train Perplexity: {train_perplexity}")
    print(f"Lambda values (λ1={lambda1}, λ2={lambda2}, λ3={lambda3}) - Development Perplexity: {dev_perplexity}")
    
    # Update best perplexity and lambda combination
    if dev_perplexity < best_dev_perplexity:
        best_dev_perplexity = dev_perplexity
        best_lambda_combination = (lambda1, lambda2, lambda3)

# Output the best lambda values and development perplexity
print(f"Best Lambda values: λ1={best_lambda_combination[0]}, λ2={best_lambda_combination[1]}, λ3={best_lambda_combination[2]}")
print(f"Best Development Perplexity: {best_dev_perplexity}")

# Calculate perplexity on the test set using the best lambda values
train_perplexity = calculate_interpolated_perplexity(
    unigram_model, bigram_model, trigram_model, train_data, *best_lambda_combination
)
test_perplexity = calculate_interpolated_perplexity(
    unigram_model, bigram_model, trigram_model, test_data, *best_lambda_combination
)

print(f"Train Perplexity with Best Lambda: {train_perplexity}")
print(f"Test Perplexity with Best Lambda: {test_perplexity}")
