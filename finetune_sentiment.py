import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import wandb
from transformers import GPT2Tokenizer
import random

from config.finetune_config import FinetuneConfig
from finetune_model import GPT2ForSentimentClassification

# Set seeds for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            # Standard cross entropy without reduction
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Apply focal loss modulation
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SentimentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, augment_positive=False):
        df = pd.read_csv(data_path)
        self.texts = df['processed_text'].tolist()
        sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.labels = [sentiment_map[s] for s in df['customer_sentiment'].tolist()]
        
        # Augment positive samples if requested
        if augment_positive:
            positive_indices = [i for i, label in enumerate(self.labels) if label == 0]
            # Add each positive sample 3 more times
            for _ in range(3):
                for idx in positive_indices:
                    self.texts.append(self.texts[idx])
                    self.labels.append(0)
        
        # Store encodings
        self.input_ids = []
        self.attention_masks = []
        
        for text in self.texts:
            encoding = tokenizer.encode_plus(
                text,
                truncation=True, 
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True
            )
            self.input_ids.append(encoding['input_ids'])
            self.attention_masks.append(encoding['attention_mask'])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    # Load configuration
    config = FinetuneConfig()
    
    wandb.init(
        project="DI725-assignment1-finetuned",
        config={
            "architecture": f"finetuned-gpt2-lstm",
            "dataset": "customer_service_sentiment",
            "learning_rate": config.learning_rate,
            "epochs": config.max_epochs,
            "batch_size": config.batch_size,
            "gamma": config.focal_loss_gamma,
            "label_smoothing": config.label_smoothing,
            "weight_decay": config.weight_decay
        }
    )
    
    # Load tokenizer for GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SentimentDataset(
        'data/sentiment/train_processed.csv', 
        tokenizer, 
        max_length=config.max_length, 
        augment_positive=True
    )
    val_dataset = SentimentDataset(
        'data/sentiment/val_processed.csv', 
        tokenizer, 
        max_length=config.max_length
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Compute class weights to handle imbalance
    y_train = np.array(train_dataset.labels)
    classes = np.unique(y_train)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # Give extra weight to positive class
    class_weights_np[0] *= 1.5
    class_weights = torch.FloatTensor(class_weights_np).to(device)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize the model
    model = GPT2ForSentimentClassification(
        model_name=config.model_name,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    model.to(device)
    
    # Count and log trainable parameters
    trainable_params = model.count_parameters()
    print(f"Model initialized with {trainable_params:,} trainable parameters")
    
    # Freeze base model if specified
    if config.freeze_base_model:
        print("Freezing base model embeddings")
        model.freeze_base_model()
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Set up learning rate scheduler with warmup
    total_steps = len(train_loader) * config.max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps
    )
    
    # Initialize focal loss with label smoothing
    focal_loss_fn = FocalLoss(
        gamma=config.focal_loss_gamma,
        weight=class_weights,
        reduction='mean',
        label_smoothing=config.label_smoothing
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    no_improvement_count = 0
    
    for epoch in range(config.max_epochs):
        # Unfreeze base model if it's time
        if config.freeze_base_model and epoch == config.unfreeze_after_epoch:
            print(f"Epoch {epoch+1}: Unfreezing base model")
            model.unfreeze_base_model()
        
        # Training phase
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            # Forward pass
            logits, _ = model(inputs, attention_mask, targets)
            loss = focal_loss_fn(logits, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {time.time()-start_time:.2f}s")
                wandb.log({"train_batch_loss": loss.item()})
                start_time = time.time()
        
        # Log epoch training stats
        avg_train_loss = total_loss / len(train_loader)
        wandb.log({"train_epoch_loss": avg_train_loss, "epoch": epoch})
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                
                logits, _ = model(inputs, attention_mask, targets)
                loss = focal_loss_fn(logits, targets)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
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
        
        # Check for improvement
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
            os.makedirs('out-sentiment-finetuned', exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join('out-sentiment-finetuned', 'gpt2_sentiment_model.pt'))
            
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join('out-sentiment-finetuned', 'tokenizer'))
            
            # Log classification report
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
            
        if no_improvement_count >= config.early_stopping_patience:
            print(f"No improvement for {config.early_stopping_patience} epochs. Early stopping.")
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()