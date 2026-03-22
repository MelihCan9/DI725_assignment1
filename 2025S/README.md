# Customer Service Sentiment Analysis — 2025S Submission

## Overview
This project implements sentiment analysis for customer service conversations using attention-based (transformer) architectures. The task is a 3-class classification problem: **positive**, **neutral**, or **negative** sentiment.

Two approaches are implemented and compared:
- A custom Transformer trained from scratch (nanoGPT-based architecture)
- A fine-tuned GPT-2 model with a classification head

All code is contained in a single reproducible Jupyter notebook.

## Dataset
The dataset consists of customer service conversations with the following characteristics:
- **Training set:** 970 samples
- **Test set:** 30 samples (balanced: 10 per class)
- **Features used:** `conversation` and `customer_sentiment` columns only
- **Sentiment distribution (train):** 55.9% neutral, 42.4% negative, 1.8% positive — highly imbalanced

## Project Structure
```plaintext
2025S/
├── DI725_assignment1_2025S.ipynb   # Main notebook (with outputs)
├── data/
│   ├── train.csv                   # Training data
│   └── test.csv                    # Test data
└── plots/
    ├── sentiment_distribution.png
    ├── complexity_vs_sentiment.png
    ├── agent_vs_sentiment.png
    ├── conv_length_distribution.png
    ├── confusion_matrix_Custom_Transformer.png
    └── confusion_matrix_GPT-2_Fine-tuned.png
```

## Preprocessing
- Only **customer turns** are extracted from conversations (agent lines dropped)
- Text is lowercased and whitespace-normalized
- Train/validation split: 80/20 stratified
- Positive class oversampling (×4) + weighted random sampler to handle class imbalance

## Model Architecture

### Custom Transformer (from scratch)
- 6-layer transformer with multi-head self-attention (nanoGPT-based)
- GPT-2 tokenizer (vocab size 50304)
- Global average pooling → 2-layer classification head with Tanh + Dropout
- ~16M parameters

### Fine-tuned GPT-2
- Pre-trained GPT-2 base model (124M parameters)
- Masked mean pooling over last hidden states
- Linear classification head
- Gradual unfreezing: head-only for first 3 epochs, then full fine-tuning

## Results on Test Set

| Model | Accuracy | F1 Weighted | F1 Macro |
|-------|----------|-------------|----------|
| Custom Transformer (scratch) | 56.7% | 0.546 | 0.546 |
| Fine-tuned GPT-2 | **76.7%** | **0.774** | **0.774** |

Fine-tuned GPT-2 significantly outperforms the from-scratch model, demonstrating the value of transfer learning especially on small datasets.

## Experiment Tracking
Experiments are tracked with Weights & Biases (WANDB).

**Project Dashboards:**
- [DI725-assignment1-2025S](https://wandb.ai/mchamurcu-metu-middle-east-technical-university/DI725-assignment1-2025S?nw=nwusermchamurcu)

> **Note:** Trained model files (.pt) are not included as they exceed 25MB.
