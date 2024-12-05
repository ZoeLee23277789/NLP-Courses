import os
from collections import defaultdict
from sklearn.metrics import classification_report

# 加載數據的函數
def load_treebank_splits(datadir):
    def load_split(subdirs):
        sentences = []
        for subdir in subdirs:
            path = os.path.join(datadir, subdir)
            if not os.path.exists(path):
                print(f"Directory {path} does not exist!")
                continue
            for file in os.listdir(path):
                if file.endswith('.pos'):
                    sentences.extend(load_pos_file(os.path.join(path, file)))
        return sentences

    train = load_split([f"{i:02d}" for i in range(0, 19)])  # Train: Folders 00-18
    dev = load_split([f"{i:02d}" for i in range(19, 22)])   # Dev: Folders 19-21
    test = load_split([f"{i:02d}" for i in range(22, 25)])  # Test: Folders 22-24

    return train, dev, test

def load_pos_file(filepath):
    sentences = []
    with open(filepath, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):  # Skip empty lines and headers
                continue
            parts = line.split()
            for part in parts:
                if "/" in part:
                    word, tag = part.rsplit("/", 1)
                    sentence.append((word, tag))
            if line.endswith("."):
                sentences.append(sentence)
                sentence = []
    return sentences
    # 建立 HMM 模型

transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
state_counts = defaultdict(int)

def build_hmm(data):
    for sentence in data:
        prev_tag = "<START>"
        for word, tag in sentence:
            transition_counts[prev_tag][tag] += 1
            emission_counts[tag][word] += 1
            state_counts[prev_tag] += 1
            state_counts[tag] += 1
            prev_tag = tag
        transition_counts[prev_tag]["<STOP>"] += 1
        state_counts[prev_tag] += 1

def calculate_probabilities(transition_counts, emission_counts, state_counts, alpha=1.0):
    transition_probs = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    emission_probs = defaultdict(lambda: defaultdict(lambda: float('-inf')))

    # 計算轉移概率
    for prev_tag in transition_counts:
        total = sum(transition_counts[prev_tag].values()) + alpha * len(state_counts)
        for curr_tag in state_counts:
            transition_probs[prev_tag][curr_tag] = \
                (transition_counts[prev_tag].get(curr_tag, 0) + alpha) / total

    for tag in emission_counts:
        total = sum(emission_counts[tag].values()) + alpha * len(emission_counts)
        for word in emission_counts[tag]:
            emission_probs[tag][word] = (emission_counts[tag][word] + alpha) / total
        # 添加 <UNK>
        emission_probs[tag]["<UNK>"] = alpha / total

    # 確保每個標籤都有 <UNK>
    for tag in state_counts:
        if "<UNK>" not in emission_probs[tag]:
            total = sum(emission_counts[tag].values()) + alpha * len(emission_counts)
            emission_probs[tag]["<UNK>"] = alpha / total

    return transition_probs, emission_probs

def viterbi(sentence, transition_probs, emission_probs, state_counts):
    n = len(sentence)
    states = list(state_counts.keys())
    dp = [{} for _ in range(n)]  # DP table
    backpointer = [{} for _ in range(n)]  # Backpointer table

    # 初始狀態
    for state in states:
        dp[0][state] = (
            transition_probs["<START>"].get(state, 1e-6) * 
            emission_probs[state].get(sentence[0], emission_probs[state].get("<UNK>", 1e-6))
        )
        backpointer[0][state] = "<START>"

    # 動態規劃
    for t in range(1, n):
        for state in states:
            max_prob, best_prev_state = max(
                (
                    dp[t - 1][prev_state] *
                    transition_probs[prev_state].get(state, 1e-6) *
                    emission_probs[state].get(sentence[t], emission_probs[state].get("<UNK>", 1e-6)),
                    prev_state
                )
                for prev_state in states
            )
            dp[t][state] = max_prob
            backpointer[t][state] = best_prev_state

    # 終止狀態
    max_prob, best_final_state = max(
        (dp[n - 1][state] * transition_probs[state].get("<STOP>", 1e-6), state)
        for state in states
    )

    # 回溯
    best_path = []
    current_state = best_final_state
    for t in range(n - 1, -1, -1):
        best_path.append(current_state)
        current_state = backpointer[t][current_state]
    best_path.reverse()

    return best_path

# 評估模型性能
def evaluate(test_set, predictions):
    y_true, y_pred = [], []
    for sentence, pred in zip(test_set, predictions):
        y_true.extend([tag for _, tag in sentence])
        y_pred.extend(pred)
    print(classification_report(y_true, y_pred))

# 主程序
datadir = r"C:\Users\USER\Downloads\NLP-Courses\NLP201\Assignments\HW3\data\penn-treeban3-wsj\wsj"
train, dev, test = load_treebank_splits(datadir)

# 使用部分資料集
train = train  # 僅使用前 100 條訓練數據
dev = dev      # 僅使用前 50 條開發數據
test = test    # 僅使用前 50 條測試數據

print(f"Train sentences: {len(train)}")
print(f"Dev sentences: {len(dev)}")
print(f"Test sentences: {len(test)}")

# 計算轉移和發射計數
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
state_counts = defaultdict(int)

for sentence in train:
    prev_tag = "<START>"
    for word, tag in sentence:
        transition_counts[prev_tag][tag] += 1
        emission_counts[tag][word] += 1
        state_counts[tag] += 1
        prev_tag = tag
    transition_counts[prev_tag]["<STOP>"] += 1

# 計算轉移和發射概率
transition_probs, emission_probs = calculate_probabilities(
    transition_counts, emission_counts, state_counts, alpha=1.0
)

# 測試 Viterbi
test_sentences = [[word for word, tag in sentence] for sentence in test]
predictions = []
for i, sentence in enumerate(test_sentences):
    try:
        prediction = viterbi(sentence, transition_probs, emission_probs, state_counts)
        predictions.append(prediction)
    except Exception as e:
        print(f"Error in sentence {i}: {sentence}")
        print(f"Error: {e}")
        predictions.append(["<UNK>"] * len(sentence))

# 評估模型
evaluate(test, predictions)
