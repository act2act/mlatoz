import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

print(words[:10])
# print(len(words))
# print(min(len(w) for w in words))
# print(max(len(w) for w in words))

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for (ch1, ch2) in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
# print(sorted(b.items(), key = lambda kv: -kv[1]))

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
# print(stoi)
# print(itos)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for (ch1, ch2) in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        N[ix1, ix2] += 1
# print(N)

# plt.figure(figsize=(10, 10))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
#         plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
# plt.axis('off')
# plt.show()

g = torch.Generator().manual_seed(2147484647)
P = (N+1).float()
P /= P.sum(1, keepdim=True)
for i in range(5):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

log_likelihood = 0.0
n = 0
for w in words:
    chars = ['.'] + list(w) + ['.']
    for (ch1, ch2) in zip(chars, chars[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += torch.log(prob)
        n += 1
"""log(a*b*c) = log(a) + log(b) + log(c)"""
print(f'{log_likelihood=}')
print(f'{-log_likelihood=}')
print(f'{-log_likelihood/n=}')

xs, ys = [], []
for w in words[:1]:
    chars = ['.'] + list(w) + ['.']
    for (ch1, ch2) in zip(chars, chars[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

g = torch.Generator().manual_seed(2147484647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W # log-counts
    counts = logits.exp() # equivalent to N matrix
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    print('loss: ', loss.item())

    # backward pass
    W.grad = None
    loss.backward()

    # update weights
    W.data += -0.1 * W.grad