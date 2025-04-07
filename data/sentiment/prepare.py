import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import wandb

# Initialize wandb
wandb.init(
    entity="mchamurcu-metu-middle-east-technical-university",
    project="DI725-assignment1",
    config={
        "step": "data_preparation",
        "dataset": "customer_service_sentiment",
    },
    name="data_exploration"
)

print("Loading and exploring the dataset...")

# Create the output directory if it doesn't exist
os.makedirs('data/sentiment/plots', exist_ok=True)

try:
    train_df = pd.read_csv('data/sentiment/train.csv')
    test_df = pd.read_csv('data/sentiment/test.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset files not found. Please place train.csv and test.csv in the data/sentiment directory.")
    exit(1)

# Basic information about the dataset
print(f"\nTraining set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Display column names - to identify the correct sentiment column
print("\nColumns in the dataset:")
print(train_df.columns.tolist())

# Display first few rows to understand the structure
print("\nFirst few rows of the dataset:")
print(train_df.head())

print("\nDataset info:")
print(train_df.info())

# Determine the correct sentiment column name
sentiment_column = 'customer_sentiment'  # Updated to match the actual column name

# Check if we have a sentiment column
if sentiment_column not in train_df.columns:
    print(f"\nWARNING: Column '{sentiment_column}' not found. Please check your dataset.")
    print("Available columns are:", train_df.columns.tolist())
    print("Please modify the script to use the correct sentiment column name.")
    exit(1)

# Check for missing values
print("\nMissing values in training set:")
print(train_df.isnull().sum())

# Display sentiment distribution
print(f"\nSentiment distribution in training set (column: {sentiment_column}):")
sentiment_counts = train_df[sentiment_column].value_counts()
print(sentiment_counts)
print(f"Sentiment distribution percentages: {sentiment_counts / len(train_df) * 100}")

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=sentiment_column, data=train_df)
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Add count labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom')
                
plt.savefig('data/sentiment/plots/sentiment_distribution.png')
wandb.log({"sentiment_distribution": wandb.Image('data/sentiment/plots/sentiment_distribution.png')})

# Determine the text column
text_column = 'conversation' 
if text_column not in train_df.columns:
    print(f"\nWARNING: Column '{text_column}' not found. Please check your dataset.")
    print("Available columns are:", train_df.columns.tolist())
    print("Please modify the script to use the correct text column name.")
    exit(1)

# Analyze text length
train_df['text_length'] = train_df[text_column].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x='text_length', hue=sentiment_column, bins=50)
plt.title('Conversation Length by Sentiment')
plt.xlabel('Character Count')
plt.savefig('data/sentiment/plots/text_length_distribution.png')
wandb.log({"text_length_distribution": wandb.Image('data/sentiment/plots/text_length_distribution.png')})

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing
print(f"\nApplying text preprocessing to column '{text_column}'...")
train_df['processed_text'] = train_df[text_column].apply(preprocess_text)
test_df['processed_text'] = test_df[text_column].apply(preprocess_text)

# Create train and validation split
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df[sentiment_column]
)

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# Save processed data
train_data.to_csv('data/sentiment/train_processed.csv', index=False)
val_data.to_csv('data/sentiment/val_processed.csv', index=False)
test_df.to_csv('data/sentiment/test_processed.csv', index=False)

print("\nProcessed files saved to:")
print("- data/sentiment/train_processed.csv")
print("- data/sentiment/val_processed.csv")
print("- data/sentiment/test_processed.csv")

# Display sample of processed data
print("\nSample of processed data:")
print(train_data[[text_column, 'processed_text', sentiment_column]].head(3))

# Log sample data to wandb
wandb.log({"sample_data": wandb.Table(dataframe=train_data[['processed_text', sentiment_column]].head(10))})

# Finish wandb run
wandb.finish()

print("\nData preparation complete!")