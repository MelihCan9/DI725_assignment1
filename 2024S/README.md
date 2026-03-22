# Customer Service Sentiment Analysis Project

## Overview

This project implements sentiment analysis models for customer service conversations. It encompasses data preparation, model training, and evaluation of multiple approaches:
- A custom transformer model built from scratch
- A fine-tuned GPT-2 model with a specialized classification head

The models classify customer sentiment as **positive**, **neutral**, or **negative** based on conversation text.

## Dataset

The dataset consists of customer service conversations with the following characteristics:
- **Training set:** 970 samples
- **Test set:** 30 samples
- **Features include:** Issue areas, categories, product information, and full conversation text
- **Sentiment distribution:** Imbalanced with 55.9% neutral, 42.4% negative, and only 1.8% positive samples

## Project Structure

```plaintext
├── data/sentiment/          # Dataset directory
│   ├── train.csv            # Original training data
│   ├── test.csv             # Test data
│   ├── prepare.py           # Data preprocessing script
│   └── plots/               # Visualizations from data exploration
├── config/                  # Configuration files
│   ├── finetune_config.py   # Configuration for fine-tuning
│   └── train_sentiment.py   # Configuration for from-scratch training
├── sentiment_model.py       # Custom transformer implementation
├── finetune_model.py        # GPT-2 fine-tuning implementation
├── train_sentiment.py       # Training script for custom model
├── finetune_sentiment.py    # Fine-tuning script for GPT-2
├── bench.py                 # Benchmarking script to compare models
└── wandb_test.py            # Simple Weights & Biases test script
```

## Model Architecture

### Custom Transformer Model
- Multi-head self-attention mechanism
- Positional encoding for sequence processing
- Multiple transformer encoder layers
- Classification head for sentiment prediction

### Fine-tuned GPT-2
- Pre-trained GPT-2 base model
- Custom classification head added for sentiment prediction
- Dropout for regularization
- Fine-tuned on the customer service dataset

## Results and Performance

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Custom Transformer | 78.3% | 0.76 | 0.79 | 0.74 |
| Fine-tuned GPT-2 | 83.7% | 0.82 | 0.84 | 0.81 |

The fine-tuned GPT-2 model outperforms the custom transformer, particularly in handling the imbalanced nature of the dataset.

## Experiment Tracking

This project uses Weights & Biases (wandb) to track experiments and visualize model performance.

**Project Dashboards:**
- [DI725-assignment1-from_scratch](https://wandb.ai/mchamurcu-metu-middle-east-technical-university/DI725-assignment1-from_scratch?nw=nwusermchamurcu)
- [DI725-assignment1-finetuned](https://wandb.ai/mchamurcu-metu-middle-east-technical-university/DI725-assignment1-finetuned?nw=nwusermchamurcu)
- [DI725-sentiment-benchmark](https://wandb.ai/mchamurcu-metu-middle-east-technical-university/DI725-sentiment-benchmark)


PS: Since model files size is bigger than 25MB the project does not include models itself.
