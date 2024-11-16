# N-gram Language Model with Linear Interpolation

This project implements an N-gram language model with unigram, bigram, and trigram components, using linear interpolation smoothing to improve performance on unseen data. Calculates perplexity on training, development, and test sets, allowing for evaluation and tuning of the model's hyperparameters to achieve optimal results.

## Features

- Unigram, Bigram, and Trigram Models: Calculate probabilities based on individual words, word pairs, and word triplets.
- Out-of-Vocabulary (OOV) Handling: Words with low frequency (appearing less than a threshold) are replaced with a special `<UNK>` token to handle unknown words effectively.
- Linear Interpolation Smoothing: A weighted linear combination of unigram, bigram, and trigram probabilities to handle sparsity and improve model generalization.
- Hyperparameter Tuning: Multiple sets of lambda values for interpolation weights are evaluated to determine the best configuration for minimizing perplexity.

## Files

- `N_gram_LM.py`: The main script implementing the N-gram language model with functions for loading data, building unigram, bigram, and trigram models, and calculating perplexity with linear interpolation.
## Dependencies

All required packages are listed in `requirements.txt`.

## Usage

1. Data Preparation: 
    - `train_data = load_data('HW2/A2-Data/1b_benchmark.train.tokens')`
    - `dev_data = load_data('HW2/A2-Data/1b_benchmark.dev.tokens')`
    - `test_data = load_data('HW2/A2-Data/1b_benchmark.test.tokens')`

2. Running the Model:

    ```bash
    python N_gram_LM.py
    ```



