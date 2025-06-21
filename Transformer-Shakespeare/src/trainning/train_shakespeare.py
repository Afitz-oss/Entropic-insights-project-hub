import os
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Setup paths
ROOT_DIR = Path(__file__).parent.parent.parent  # Get project root
DATA_DIR = ROOT_DIR / 'data' / 'Subword-level-Token'
OUT_DIR = ROOT_DIR / 'out'
OUT_DIR.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

# I/O
out_dir = str(OUT_DIR)
model_path = OUT_DIR / 'ckpt.pt'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 40
batch_size = 12
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Initialization
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
def get_batch(split):
    """Load data from binary files with path checking."""
    data_file = DATA_DIR / f'{split}.bin'
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find {split}.bin in {DATA_DIR}. "
            "Please run the tokenization script first: "
            "python data/Subword-level-Token/NN-Shakes-Token-Gen-Subword.py"
        )
    
    data = np.memmap(str(data_file), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

def init_training():
    """Check environment and data before training."""
    # Check CUDA
    if not torch.cuda.is_available() and device == 'cuda':
        raise RuntimeError("CUDA is not available but device='cuda' was specified")
    
    # Check data files
    required_files = ['train.bin', 'val.bin']
    missing_files = [f for f in required_files if not (DATA_DIR / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing data files: {missing_files}\n"
            f"Expected location: {DATA_DIR}\n"
            "Please run tokenization script first"
        )
    
    # Print training config
    print("Training Configuration:")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Device: {device} ({device_type})")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterations: {max_iters}")

# Model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                 bias=bias, vocab_size=50257, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Initialize a GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# Learning rate decay
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    """Helper function to evaluate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
iter_num = 0
best_val_loss = 1e9
X, Y = get_batch('train')
t0 = time.time()
while True:
    # Determine learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Forward backward update
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X_next, Y_next = get_batch('train')
        scaler.scale(loss).backward()
        
        if (micro_step + 1) == gradient_accumulation_steps:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        X, Y = X_next, Y_next

    # Logging with timing
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item()*gradient_accumulation_steps:.4f}, time {dt*1000:.2f}ms, lr {lr:e}")

    iter_num += 1
    if iter_num > max_iters:
        break

if __name__ == '__main__':
    try:
        # Initialize and check environment
        init_training()
        
        # Start training
        print("\nStarting training...")
        model = train_model()
        
        print(f"\nTraining completed! Model saved to {model_path}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

