import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 300
LR = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
EVAL_ITERS = 200
N_EMBD = 32

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Tokenisation - Create a mapping from chars to ints
s_to_i = { ch:i for i,ch in enumerate(chars) }
i_to_s = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [s_to_i[c] for c in s] # encoder: take a string, output a list of ints
decode = lambda l: ''.join([i_to_s[i] for i in l]) #decoder: take a list of integers, output a string

# train test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    index_list = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) # 4 random indices from training set
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in index_list])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in index_list])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    # One head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # still (B, T, T)

        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    # multiple attnetion heads in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.sa_heads = MultiHeadAttention(4, N_EMBD//4) # sa self attention
        self.lm_head = nn.Linear(N_EMBD, vocab_size) # lm language model


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both BxT tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x)
        logits = self.lm_head(x) # (B, T, vocab_size))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]

            # get preds
            logits, loss = self(idx_cond) # = self.forward(idx_cond)

            # focus only on last timestep
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# only the optimizer model is on the gpu??
optimizer = torch.optim.AdamW(m.parameters(), lr=LR)

for iter in range(MAX_ITERS):
    # every once in a while evaluate loss on train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {iter:4.0f} - train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))