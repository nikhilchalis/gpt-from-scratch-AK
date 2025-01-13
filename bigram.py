import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LR = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
EVAL_ITERS = 200

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both BxT tensor of integers
        logits = self.token_embedding_table(idx) #BxTxC

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
            # get preds
            logits, loss = self(idx) # = self.forward(idx)

            # focus only on last timestep
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
model = BigramLanguageModel(vocab_size)
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