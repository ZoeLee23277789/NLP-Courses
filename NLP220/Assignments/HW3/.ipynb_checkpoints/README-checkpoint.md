# Hist Gradient Boosting Model with Optuna Optimization

This project trains a multi-label classification model using `HistGradientBoostingClassifier` with the best hyperparameters obtained via Optuna optimization. The model classifies the topics of arXiv paper abstracts based on TF-IDF features.

## Project Structure
- `arxiv_data.json`: Dataset containing titles, abstracts, and labels.
- `main.py`: Main script that preprocesses the data, trains the model with the optimal hyperparameters, evaluates it, and generates classification reports.
- `Optimized_HistGradientBoosting_Results.txt`: File containing the validation and test classification reports.

## Best Hyperparameters
After running Optuna, the best hyperparameters found were:
- `max_iter`: 291
- `learning_rate`: 0.0235
- `max_leaf_nodes`: 96
- `min_samples_leaf`: 9

These values are directly used in `HistGradientBoostingClassifier` to train and evaluate the model.

## Files
- `main.py`: The main script for running the project.
- `arxiv_data.json`: Dataset file with abstracts and labels for multi-label classification.
- `Optimized_HistGradientBoosting_Results.txt`: Generated file containing model evaluation reports on validation and test sets.

## Usage

### Prerequisites
Install the required libraries:
```bash
pip install -r requirements.txt
