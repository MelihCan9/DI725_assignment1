"""
A much shorter version of train.py for benchmarking sentiment classification models
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Import the model classes
from finetune_model import GPT2ForSentimentClassification
from sentiment_model import SentimentTransformer

# -----------------------------------------------------------------------------
batch_size = 16
max_length = 128
bias = False
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Set to False to avoid compatibility issues
profile = True  
test_file = 'data/sentiment/test_processed.csv' 
model_paths = {
    'gpt2-finetuned': 'out-sentiment-finetuned/gpt2_sentiment_model.pt',
    'original': 'out-sentiment/sentiment_model.pt'  
}
use_wandb = False 
wandb_project = "DI725-sentiment-benchmark"
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Initialize wandb
if use_wandb:
    wandb.init(
        project=wandb_project,
        config={
            "batch_size": batch_size,
            "max_length": max_length,
            "device": device,
            "dtype": dtype,
            "compile": compile,
            "profile": profile,
            "test_file": test_file,
            "models": list(model_paths.keys())
        }
    )

# Set random seeds
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Sentiment dataset class for benchmarking
class SentimentBenchmarkDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        # Load and process the data
        df = pd.read_csv(data_path)
        self.texts = df['processed_text'].tolist()
        sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.labels = [sentiment_map[s] for s in df['customer_sentiment'].tolist()]
        
        # Store encodings
        self.encodings = []
        for text in self.texts:
            encoded = tokenizer.encode_plus(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Function to plot and log confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['positive', 'neutral', 'negative'],
                yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save locally and log to wandb
    cm_filename = f'confusion_matrix_{model_name}.png'
    plt.savefig(cm_filename)
    if use_wandb:
        wandb.log({f"confusion_matrix_{model_name}": wandb.Image(cm_filename)})
    plt.close()

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load benchmark dataset
benchmark_dataset = SentimentBenchmarkDataset(test_file, tokenizer, max_length)
dataloader = DataLoader(benchmark_dataset, batch_size=batch_size, shuffle=False)
print(f"Loaded {len(benchmark_dataset)} samples for benchmarking")

# Benchmarking results
results = {}

# Run benchmark for each model
for model_name, model_path in model_paths.items():
    print(f"\nBenchmarking {model_name}...")
    
    # Initialize the model based on model type
    if model_name == 'gpt2-finetuned':
        model = GPT2ForSentimentClassification(
            model_name='gpt2',  # Base model name
            num_classes=3,
            dropout=0.1
        )
    elif model_name == 'original':
        # Create a config object for the SentimentTransformer
        from sentiment_model import SentimentConfig
        config = SentimentConfig()
        config.vocab_size = 50304  # Match the saved model's vocab size
        config.block_size = 256    # Match the saved model's block size
        config.n_layer = 6         # Match the saved model's number of layers
        config.n_head = 6          # Assuming 6 heads based on standard architecture
        config.n_embd = 384        # Match the saved model's embedding dimension
        config.dropout = 0.1
        config.bias = True         # The saved model has bias terms
        config.num_classes = 3
        
        model = SentimentTransformer(config)
    
    # Load saved weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        continue
    
    model.to(device)
    
    if compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Run profiling or simple benchmarking
    batch_times = []
    all_preds = []
    all_targets = []
    all_logits = []
    
    if profile:
        # Profiling with PyTorch profiler
        wait, warmup, active = 5, 5, 5
        num_steps = wait + warmup + active
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./bench_log_{model_name}'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        ) as prof:
            
            model.eval()
            with torch.no_grad():
                for k, batch in enumerate(dataloader):
                    if k >= num_steps:
                        break
                    
                    with ctx:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        # Time each batch
                        torch.cuda.synchronize() if 'cuda' in device else None
                        start_time = time.time()
                        
                        # Forward pass
                        if model_name == 'gpt2-finetuned':
                            logits, loss = model(input_ids, attention_mask, labels)
                        elif model_name == 'original':
                            logits, loss = model(input_ids, labels)
                        
                        torch.cuda.synchronize() if 'cuda' in device else None
                        end_time = time.time()
                        batch_time = end_time - start_time
                        batch_times.append(batch_time)
                        
                        # Get predictions
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_targets.extend(labels.cpu().numpy())
                        all_logits.extend(logits.cpu().numpy())
                    
                    print(f"{k}/{num_steps} samples processed")
                    prof.step()
    
    else:
        # Simple benchmarking
        model.eval()
        
        torch.cuda.synchronize() if 'cuda' in device else None
        for stage, num_batches in enumerate([5, 10]):  # warmup, then benchmark
            t0 = time.time()
            
            with torch.no_grad():
                for k, batch in enumerate(dataloader):
                    if k >= num_batches:
                        break
                    
                    with ctx:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        # Time each batch in the benchmark stage
                        if stage == 1:
                            torch.cuda.synchronize() if 'cuda' in device else None
                            batch_start = time.time()
                        
                        # Forward pass
                        if model_name == 'gpt2-finetuned':
                            logits, loss = model(input_ids, attention_mask, labels)
                        elif model_name == 'original':
                            logits, loss = model(input_ids, labels)
                        
                        # Record batch time and collect predictions 
                        if stage == 1:
                            torch.cuda.synchronize() if 'cuda' in device else None
                            batch_end = time.time()
                            batch_times.append(batch_end - batch_start)
                            
                            # Collect predictions
                            preds = torch.argmax(logits, dim=1).cpu().numpy()
                            all_preds.extend(preds)
                            all_targets.extend(labels.cpu().numpy())
                            all_logits.extend(logits.cpu().numpy())
                    
                    print(f"{k}/{num_batches} batches processed")
            
            torch.cuda.synchronize() if 'cuda' in device else None
            t1 = time.time()
            dt = t1 - t0
            
            # Report time taken for the second stage (actual benchmark)
            if stage == 1:
                samples_processed = num_batches * batch_size
                time_per_batch = dt / num_batches * 1000  # ms
                time_per_sample = dt / samples_processed * 1000  # ms
                samples_per_second = samples_processed / dt
                
                print(f"Time per batch: {time_per_batch:.4f}ms")
                print(f"Time per sample: {time_per_sample:.4f}ms")
                print(f"Samples per second: {samples_per_second:.2f}")
                
                # Log timing metrics to wandb
                if use_wandb:
                    wandb.log({
                        f"{model_name}_time_per_batch_ms": time_per_batch,
                        f"{model_name}_time_per_sample_ms": time_per_sample,
                        f"{model_name}_samples_per_second": samples_per_second
                    })
    
    # Calculate and report metrics
    if len(all_preds) > 0 and len(all_targets) > 0:
        accuracy = accuracy_score(all_targets, all_preds)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        f1_macro = f1_score(all_targets, all_preds, average='macro')

        # Calculate per-class metrics
        report = classification_report(all_targets, all_preds,
                                      target_names=['positive', 'neutral', 'negative'],
                                      output_dict=True)
        positive_f1 = report['positive']['f1-score'] # Extract positive F1

        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")

        # Plot and log confusion matrix
        plot_confusion_matrix(all_targets, all_preds, model_name)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=['positive', 'neutral', 'negative']))

        # Calculate average batch time
        avg_batch_time = np.mean(batch_times) * 1000  # convert to ms
        avg_sample_time = avg_batch_time / batch_size
        avg_samples_per_second = batch_size / (np.mean(batch_times))

        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'positive_f1': positive_f1, # Store positive F1
            'time_per_batch_ms': avg_batch_time,
            'time_per_sample_ms': avg_sample_time,
            'samples_per_second': avg_samples_per_second
        }
        
        # Log metrics to wandb
        if use_wandb:
            metrics = {
                f"{model_name}_accuracy": accuracy,
                f"{model_name}_f1_weighted": f1_weighted,
                f"{model_name}_f1_macro": f1_macro,
                f"{model_name}_positive_f1": positive_f1,
                f"{model_name}_time_per_batch_ms": avg_batch_time,
                f"{model_name}_time_per_sample_ms": avg_sample_time,
                f"{model_name}_samples_per_second": avg_samples_per_second
            }
            
            # Add per-class metrics
            for cls in ['positive', 'neutral', 'negative']:
                if cls in report:
                    metrics[f"{model_name}_{cls}_precision"] = report[cls]['precision']
                    metrics[f"{model_name}_{cls}_recall"] = report[cls]['recall']
                    metrics[f"{model_name}_{cls}_f1"] = report[cls]['f1-score']
            
            wandb.log(metrics)

# # Print final comparative results
# if len(results) > 1:
#     print("\nComparative Results:")
#     print("-" * 50)
    
#     # Create comparison table data
#     table_data = []
#     for model_name, metrics in results.items():
#         model_row = [
#             model_name,
#             f"{metrics['accuracy']:.4f}",
#             f"{metrics['f1_weighted']:.4f}",
#             f"{metrics['f1_macro']:.4f}",
#             f"{metrics['time_per_batch_ms']:.2f}",
#             f"{metrics['time_per_sample_ms']:.2f}",
#             f"{metrics['samples_per_second']:.2f}"
#         ]
#         table_data.append(model_row)
        
#         print(f"{model_name}:")
#         print(f"  Accuracy: {metrics['accuracy']:.4f}")
#         print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
#         print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
#         print(f"  Time per batch: {metrics['time_per_batch_ms']:.2f} ms")
#         print(f"  Time per sample: {metrics['time_per_sample_ms']:.2f} ms")
#         print(f"  Samples per second: {metrics['samples_per_second']:.2f}")
#         print("-" * 50)
    
#     # Log comparison table to wandb
#     if use_wandb:
#         wandb.log({"comparison_table": wandb.Table(
#             columns=["Model", "Accuracy", "F1 (weighted)", "F1 (macro)", 
#                     "Time/Batch (ms)", "Time/Sample (ms)", "Samples/sec"],
#             data=table_data
#         )})
        
#         # Create comparison bar charts
#         for metric in ['accuracy', 'f1_weighted', 'f1_macro', 'samples_per_second']:
#             plt.figure(figsize=(10, 6))
#             model_names = list(results.keys())
#             metric_values = [results[model]['accuracy'] if metric == 'accuracy' else 
#                             results[model]['f1_weighted'] if metric == 'f1_weighted' else
#                             results[model]['f1_macro'] if metric == 'f1_macro' else
#                             results[model]['samples_per_second'] for model in model_names]
            
#             plt.bar(model_names, metric_values)
#             plt.xlabel('Model')
#             plt.ylabel(metric.replace('_', ' ').title())
#             plt.title(f'Comparison of {metric.replace("_", " ").title()}')
#             plt.tight_layout()
            
#             chart_filename = f'comparison_{metric}.png'
#             plt.savefig(chart_filename)
#             if use_wandb:
#                 wandb.log({f"comparison_{metric}": wandb.Image(chart_filename)})
#             plt.close()

# Print the final comparison table
print("\n\n--- Final Benchmark Comparison ---")
print("| Model             | Test Accuracy | Test Weighted F1 | Test Macro F1 | Positive Class F1 |")
print("| :---------------- | :------------ | :--------------- | :------------ | :---------------- |")


model_name_map = {
    'original': 'From-Scratch',
    'gpt2-finetuned': 'Fine-tuned GPT-2'
    # Add other models here if benchmarked
}

for model_name, metrics in results.items():
    display_name = model_name_map.get(model_name, model_name) # Use mapped name or original
    print(f"| {display_name:<17} | {metrics['accuracy']:.4f}        | {metrics['f1_weighted']:.4f}           | {metrics['f1_macro']:.4f}        | {metrics['positive_f1']:.4f}              |")

print("-" * 30) # Separator

if use_wandb:
    wandb.finish()

print("Benchmarking completed successfully!")
