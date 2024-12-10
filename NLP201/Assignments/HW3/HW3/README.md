# HMM POS Tagger with Viterbi Decoding

This project implements a Part-of-Speech (POS) tagging system using a Hidden Markov Model (HMM) and the Viterbi decoding algorithm. The implementation is based on additive smoothing to handle rare events and introduces the `<UNK>` token for handling unseen words.

## Project Structure

- **HMM_Viterbi.py**: The main script to load data, train the HMM model, and evaluate results using the Viterbi algorithm.
- **data/**: Folder containing the dataset split into train, dev, and test sets.


## Prerequisites

Ensure the following dependencies are installed:
- Python 3.x
- NumPy
- SciKit-Learn

You can install these using the following command:
```bash
pip install -r requirements.txt

