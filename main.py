import torch
import flash_attn

torch.manual_seed(0)


# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
n_head = 1
seq_len = 64 
head_embd = 64 

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

# q = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float).reshape(1, 1, 4, 4).cuda()
# k = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float).reshape(1, 1, 4, 4).cuda()
# v = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float).reshape(1, 1, 4, 4).cuda()


o = flash_attn.forward(q, k, v)

print(o)
print(o.size());


import torch.nn.functional as F

print('======QKV======')
# print(q)
# print(k)
# print(v)
s = torch.matmul(q, k.transpose(-2, -1))
# print(s)
p = F.softmax(s, dim=-1)
# print(p.size())
ans = torch.matmul(p, v)
print(ans)

are_close = torch.allclose(o, ans, atol=1e-2)  # `atol` is the absolute tolerance
assert are_close, "Testcase is wrong"

