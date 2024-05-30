import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum length for prediction?
max_iters = 1000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
# ----------------

torch.manual_seed(1337) # for reproducibility

# Load the text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Overview of the text
# print('length of text: ', len(text))
# print(text[:100])

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print('vocab size: ', vocab_size)

# Tokenize the text
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode('hii there'))
# print(decode(encode('hii there')))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data[:100])

# Split up the data into train and validation sets
n = int(0.9*(len(data)))
train_data, val_data = data[:n], data[n:]

# Chunk the data into batches
# block_size = 8
#
# x = train_data[:block_size+1]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f'when input is {context} the target: {target}')

# Take a concern of the batches
# batch_size = 4
# block_size = 8

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f'when input is {context} the target: {target}')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create a simple bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # get the predictions
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=-1) # append sampled index to the running sequence
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

