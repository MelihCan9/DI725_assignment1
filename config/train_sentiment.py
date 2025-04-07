# Config for training a sentiment analysis model from scratch

# Data and model paths
out_dir = 'out-sentiment'
eval_interval = 50
log_interval = 10
eval_iters = 20
always_save_checkpoint = True

# Model parameters
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2
bias = True

# Data parameters
batch_size = 32
block_size = 128
num_classes = 3  # positive, neutral, negative

# Optimizer parameters
learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-5

# Sentiment-specific parameters
dataset = 'sentiment'
gradient_accumulation_steps = 1  # Used for larger models or smaller batch sizes