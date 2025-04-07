
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sentiment_model import SentimentConfig, SentimentTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import wandb
from transformers import GPT2Tokenizer
import random

# Set torch random seed for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

class SentimentDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, max_length=128, augment_positive=False):
        df = pd.read_csv(data_path)
        self.texts = df['processed_text'].tolist()
        sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.labels = [sentiment_map[s] for s in df['customer_sentiment'].tolist()]
        
        # Augment positive samples if requested
        if augment_positive:
            positive_indices = [i for i, label in enumerate(self.labels) if label == 0]
            # Simple repetition augmentation for positive class
            for _ in range(3):  # Add each positive sample 3 more times
                for idx in positive_indices:
                    self.texts.append(self.texts[idx])
                    self.labels.append(0)
        
        self.encodings = []
        if tokenizer is not None:
            for text in self.texts:
                encoded = tokenizer.encode(text, truncation=True, max_length=max_length, padding='max_length')
                self.encodings.append(encoded)
        else:
            for text in self.texts:
                tokens = [ord(c) for c in text[:max_length]]
                if len(tokens) < max_length:
                    tokens = tokens + [0] * (max_length - len(tokens))
                self.encodings.append(tokens)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def focal_loss_with_smoothing(logits, targets, class_weights, gamma=2.0, smoothing=0.1):
    num_classes = logits.size(-1)
    
    # Create smoothed labels
    with torch.no_grad():
        smooth_targets = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = smooth_targets * (1 - smoothing) + smoothing / num_classes
    
    # Apply softmax and get log probabilities
    log_probs = F.log_softmax(logits, dim=1)
    
    # Calculate cross entropy with smoothed targets
    loss = -(smooth_targets * log_probs).sum(dim=1)
    
    # Apply focal weighting
    pt = torch.exp(-loss)
    focal_weight = (1 - pt) ** gamma
    
    # Apply class weights
    sample_weights = class_weights[targets]
    
    return (focal_weight * loss * sample_weights).mean()

def main():
    wandb.init(
        project="DI725-assignment1",
        config={
            "learning_rate": 5e-5,  # Reduced learning rate
            "architecture": "transformer",
            "dataset": "customer_service_sentiment",
            "epochs": 15,
            "batch_size": 16,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
            "positive_augmentation": True
        }
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    config = SentimentConfig()  # Uses dropout (0.3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SentimentTransformer(config)
    model.to(device)
    print(f"Model initialized with {model.get_num_params():,} parameters")
    
    # Add positive sample augmentation
    train_dataset = SentimentDataset('data/sentiment/train_processed.csv', 
                                    tokenizer=tokenizer, 
                                    max_length=config.block_size,
                                    augment_positive=True)
    val_dataset = SentimentDataset('data/sentiment/val_processed.csv', 
                                   tokenizer=tokenizer, 
                                   max_length=config.block_size)
    
    y_train = np.array(train_dataset.labels)
    classes = np.unique(y_train)
    
    # Enhanced class weights with even more emphasis on positive class
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # Give extra weight to positive class
    class_weights_np[0] *= 1.5  
    class_weights = torch.FloatTensor(class_weights_np).to(device)
    print(f"Class weights: {class_weights}")
    
    sample_weights = [class_weights_np[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Improved optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=5e-5,  # Smaller learning rate
                                 weight_decay=0.01)  # Add weight decay
    
    # Learning rate scheduler with warmup
    num_epochs = 15
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                  float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_f1 = 0
    no_improvement_count = 0
    max_no_improvement = 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            logits, _ = model(inputs, targets)
            # Use improved loss function with label smoothing
            loss = focal_loss_with_smoothing(logits, targets, class_weights, gamma=2.0, smoothing=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # Update learning rate per step
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {time.time()-start_time:.2f}s")
                wandb.log({
                    "train_batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                start_time = time.time()
        
        avg_train_loss = total_loss / len(train_loader)
        wandb.log({"train_epoch_loss": avg_train_loss, "epoch": epoch})
        
        # Validation loop
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                
                logits, _ = model(inputs, targets)
                # Use same loss function but without smoothing for validation
                loss = focal_loss_with_smoothing(logits, targets, class_weights, gamma=2.0, smoothing=0.0)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        
        print(f"Epoch {epoch+1} completed. Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Macro F1: {f1_macro:.4f}")
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_f1": f1,
            "val_f1_macro": f1_macro,
            "epoch": epoch
        })
        
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
            improved = True
            
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            print(f"New best macro F1 score: {best_val_f1:.4f}")
            improved = True
        
        if improved:
            no_improvement_count = 0
            os.makedirs('out-sentiment', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('out-sentiment', 'sentiment_model.pt'))
            
            print("\nClassification Report:")
            target_names = ['positive', 'neutral', 'negative']
            report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
            print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))
            
            for label in target_names:
                if label in report:
                    wandb.log({
                        f"{label}_precision": report[label]['precision'],
                        f"{label}_recall": report[label]['recall'],
                        f"{label}_f1": report[label]['f1-score']
                    })
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= max_no_improvement:
            print(f"No improvement for {max_no_improvement} epochs. Early stopping.")
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()