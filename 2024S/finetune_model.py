import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel, GPT2Model

class GPT2ForSentimentClassification(nn.Module):
    """GPT-2 model fine-tuned for sentiment classification"""
    
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        # Load pre-trained GPT-2 model using standard method
        self.transformer = GPT2Model.from_pretrained(model_name)
        config = self.transformer.config
        
        # Add classification head on top of GPT-2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.n_embd, num_classes)
        
        # Initialize the classification head
        with torch.no_grad():
            self.classifier.weight.normal_(mean=0.0, std=0.02)
            self.classifier.bias.zero_()
    
    def forward(self, input_ids, attention_mask=None, targets=None):
        # Process through GPT-2 transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the hidden states
        hidden_states = outputs[0]  # Last layer hidden states
        
        # Mean pooling - take attention mask into account for averaging
        if attention_mask is not None:
            # Expand attention mask to match hidden states dimensions 
            # [batch_size, seq_length, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Apply mask and compute mean
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple mean pooling if no attention mask
            pooled_output = hidden_states.mean(dim=1)
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def freeze_base_model(self):
        """Freeze all parameters of the base GPT-2 model"""
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze all parameters for full fine-tuning"""
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)