import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 12
BLOCK_SIZE = 65  # Maximum sequence length
LEARNING_RATE = 3e-4
MAX_EPOCHS = 5
EVAL_INTERVAL = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BinaryDataset(Dataset):
    def __init__(self, data_path):
        # Load binary data
        self.data = np.fromfile(data_path, dtype=np.uint8)
        self.data = torch.from_numpy(self.data.astype(np.int64))

    def __len__(self):
        return len(self.data) - BLOCK_SIZE

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + BLOCK_SIZE + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def train_model():
    # Initialize datasets
    train_dataset = BinaryDataset(r'c:\Users\btofi\Entropic-insights-project-hub\Transformer-Shakespeare\data\train.bin')
    val_dataset = BinaryDataset(r'c:\Users\btofi\Entropic-insights-project-hub\Transformer-Shakespeare\data\val.bin')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model parameters
    vocab_size = 256  # For byte-level encoding
    n_embd = 128     # Embedding dimension
    n_head = 4       # Number of attention heads
    n_layer = 4      # Number of transformer blocks
    dropout = 0.0    # Dropout rate

    # Initialize model
    model = GPTLanguageModel(vocab_size, n_embd, BLOCK_SIZE, n_head, n_layer, dropout)
    model = model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # Validation and saving
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, loss = model(x, y)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f"Validation loss: {val_loss:.4f}")

        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': loss.item(),
            }, 'best_model.pt')

        # Regular checkpoint every EVAL_INTERVAL epochs
        if epoch % EVAL_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': loss.item(),
            }, f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    torch.save({
        'epoch': MAX_EPOCHS-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': loss.item(),
    }, 'final_model.pt')

if __name__ == '__main__':
    train_model()