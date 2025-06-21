import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b, t, v = logits.size()
            logits = logits.view(b*t, v)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # validate that we considered every parameter
        param_dict_keys = set(param_dict.keys())
        decay_names = set(n for n, p in param_dict.items() if p.dim() >= 2)
        nodecay_names = set(n for n, p in param_dict.items() if p.dim() < 2)
        assert len(param_dict_keys - (decay_names | nodecay_names)) == 0, "parameters that were not categorized"
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        # Print parameter group info
        print(f"Number of parameters with weight decay: {len(decay_params)}")
        print(f"Number of parameters without weight decay: {len(nodecay_params)}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def train(
    # Model parameters
    n_layer=4,
    n_head=6,
    n_embd=256,
    block_size=512,
    dropout=0.3,
    bias=False,
    vocab_size=50257,
    
    # Training parameters
    batch_size=8,
    max_iters=50000,
    learning_rate=2e-4,
    weight_decay=0.3,
    beta1=0.9,
    beta2=0.95,
    grad_clip=0.5,
    
    # Early stopping parameters
    patience=10,
    min_improvement=1e-4,
    
    # Learning rate decay
    decay_lr=True,
    warmup_iters=1000,
    lr_decay_iters=50000,
    min_lr=3e-5,
    
    # Data paths
    train_data_path=None,
    val_data_path=None,
    
    # Output directory
    out_dir='out',
    
    # Evaluation settings
    eval_interval=250,
    eval_iters=100,
    
    # System settings
    device='cuda',
    dtype='float16',
    compile=True
):
    """
    Train a GPT model on custom data.
    """
    
    # Add data path validation
    if train_data_path is None or val_data_path is None:
        raise ValueError("Both train_data_path and val_data_path must be provided")
    
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data file not found at {train_data_path}")
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data file not found at {val_data_path}")
    
    # System setup
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model
    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout
    )
    
    model = GPT(config)
    model.to(device)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Compile model if requested, with error handling
    if compile:
        try:
            print("Attempting to compile model...")
            model = torch.compile(model)
            print("Model compilation successful!")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation...")
            # Suppress future PyTorch compilation errors
            torch._dynamo.config.suppress_errors = True
    
    # Data loading functions
    def get_batch(split):
        data_path = train_data_path if split == 'train' else val_data_path
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    # Learning rate scheduler
    def get_lr(iter):
        # Linear warmup for warmup_iters steps
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters
        # If iter > lr_decay_iters, return min learning rate
        if iter > lr_decay_iters:
            return min_lr
        # In between, use cosine decay down to min learning rate
        decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    # Test data loading before starting training
    try:
        print("Testing data loading...")
        X, Y = get_batch('train')
        print(f"Training data loaded successfully! Shape: {X.shape}")
        X, Y = get_batch('val')
        print(f"Validation data loaded successfully! Shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Initialize lists to store losses for plotting
    train_losses = []
    val_losses = []
    iterations = []
    
    def plot_losses():
        """Plot training and validation losses"""
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, label='Training Loss')
        plt.plot(iterations, val_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale for better visualization
        plt.savefig(os.path.join(out_dir, 'training_progress.png'))
        plt.show()
    
    # Training loop
    print("Beginning training...")
    iter_num = 0
    no_improvement_count = 0
    best_val_loss = float('inf')
    t0 = time.time()  # Initialize timing
    local_iter_num = 0  # Initialize local iteration counter
    
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the model
        if iter_num % eval_interval == 0:
            model.eval()
            val_losses_batch = torch.zeros(eval_iters)
            train_losses_batch = torch.zeros(eval_iters)
            
            # Evaluate on validation set
            for k in range(eval_iters):
                X, Y = get_batch('val')
                with ctx:
                    logits, loss = model(X, Y)
                val_losses_batch[k] = loss.item()
            
            # Evaluate on training set
            for k in range(eval_iters):
                X, Y = get_batch('train')
                with ctx:
                    logits, loss = model(X, Y)
                train_losses_batch[k] = loss.item()
            
            val_loss = val_losses_batch.mean()
            train_loss = train_losses_batch.mean()
            
            # Store losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            iterations.append(iter_num)
            
            # Early stopping check
            if val_loss < (best_val_loss - min_improvement):
                best_val_loss = val_loss
                no_improvement_count = 0
                # Save best model
                print(f"Saving checkpoint to {out_dir}")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'iterations': iterations,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {iter_num} iterations")
                break
            
            # Plot progress
            plot_losses()
            
            model.train()
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {lr:e}")
        
        # Forward backward update, with gradient scaling if training in fp16
        X, Y = get_batch('train')
        model.train()
        with ctx:
            logits, loss = model(X, Y)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        # Clip gradient norm
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Log training progress
        if iter_num % 100 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num > 0:
                print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:e}, {dt*1000/100:.2f}ms/iter")
            else:
                print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:e}")
        
        iter_num += 1
        
        # Termination conditions
        if iter_num > max_iters:
            break
    
    # Load best model before returning
    best_checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'))
    model.load_state_dict(best_checkpoint['model'])
    
    # Final plot at the end of training
    plot_losses()
    
    print("Training completed!")
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 'iterations': iterations}

if __name__ == "__main__":
    model, history = train(
        # Use Colab paths
        train_data_path='/content/train.bin',
        val_data_path='/content/val.bin',
        max_iters=50000,
        out_dir='/content/out/shakespeare',
        eval_interval=250,
        # Reduced model size for faster training
        n_layer=4,
        n_head=6,
        n_embd=256,
        block_size=512,
        batch_size=8,
        # Set compile to False if you're having issues
        compile=False  # Changed this to False to avoid compilation issues
    )
    
    # You can access the loss history after training
    print("Final training loss:", history['train_losses'][-1])
    print("Final validation loss:", history['val_losses'][-1])