# EA-MT Entity-Aware Machine Translation

This project demonstrates EA-MT Entity-Aware Machine Translation for English-to-Chinese translation using the IWSLT 2017 dataset and evaluates the results using BLEU, METEOR, and BERTScore metrics.

## Prerequisites

Ensure you have Python 3.7+ and install required dependencies using:

```bash
pip install transformers torch spacy tqdm evaluate bert-score nltk datasets OpenHowNet jieba
```

Download the necessary SpaCy model and OpenHowNet resources:

```bash
python -m spacy download en_core_web_trf
```

## Steps

### 1. Fine-Tune the mBART Model

1. Load and preprocess the IWSLT 2017 dataset.
2. Fine-tune the `facebook/mbart-large-50-many-to-many-mmt` model for 3 epochs.
3. Save the fine-tuned model.

### 2. Evaluate the Model

- BLEU (overlap-based metric).
- METEOR (synonym-aware alignment).
- BERTScore (contextual similarity).

## Outputs

- Fine-tuned model: `/content/final_model`.
- Evaluation results: `/content/translated_results.json`.

## Example Output

```json
{
  "Overall BLEU": 0.2431,
  "Overall METEOR": 0.4567,
  "Overall BERTScore Precision": 0.8776,
  "Overall BERTScore Recall": 0.8654,
  "Overall BERTScore F1": 0.8714
}
```
