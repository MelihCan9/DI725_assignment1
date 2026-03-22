class FinetuneConfig:
    """Configuration for GPT-2 fine-tuning on sentiment analysis"""
    
    # Model configuration
    model_name = "gpt2"  # Standard GPT-2 model
    num_classes = 3      # positive, neutral, negative
    max_length = 128
    
    # Training configuration
    batch_size = 16
    learning_rate = 3e-5
    classifier_learning_rate = 1e-4
    weight_decay = 0.01
    max_epochs = 10
    warmup_ratio = 0.1
    
    # Regularization and optimization
    dropout = 0.1
    label_smoothing = 0.1
    focal_loss_gamma = 2.0
    gradient_clip = 1.0
    
    # Fine-tuning strategy
    freeze_base_model = True
    unfreeze_after_epoch = 2
    
    # Early stopping
    early_stopping_patience = 5