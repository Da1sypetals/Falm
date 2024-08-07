import torch
import flash_attn


# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
n_head = 1
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

o = flash_attn.forward(q, k, v)

print(o)

# O = flash_attn.backward(Q, K, V);
# print(O)
# print(O.size());
