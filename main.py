import torch
import flash_attn
import torch.nn.functional as F

torch.manual_seed(0)

def test_forward():
    batch_size = 10
    n_head = 1
    seq_len = 400 
    head_embd = 4 

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    print('======Attention======')
    s = torch.matmul(q, k.transpose(-2, -1))
    p = F.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    # print(o)

    print('======Flash Attention======')
    flash_o, flash_l, flash_m = flash_attn.forward(q, k, v)
    # print(flash_o)
    # print(flash_l)
    # print(flash_m)

    are_close = torch.allclose(o, flash_o, atol=1e-2)  # `atol` is the absolute tolerance
    assert are_close, "foward is wrong"

def test_backward():
    batch_size = 5 
    n_head = 2
    seq_len = 6 
    head_embed = 2

    q = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()

    print('======Attention======')
    s = torch.matmul(q, k.transpose(-2, -1))
    m = torch.max(s, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(s - m)
    l = torch.sum(exp_scores, dim=-1, keepdim=True)
    p = exp_scores / l
    o = torch.matmul(p, v)
    
    #random init o's gradient
    do = torch.randn_like(o).cuda()

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    p.retain_grad()
    s.retain_grad()

    o.backward(do)

    dq = q.grad
    dk = k.grad
    dv = v.grad

    # print("dP", p.grad)
    # print("dS", s.grad)  
    # print("dQ:", dq)
    # print("dK:", dk)
    # print("dV:", dv)

    print('======Flash Attention======')
    flash_dq, flash_dk, flash_dv = flash_attn.backward(q, k, v, o, do, l, m)
    # print("dQ:", flash_dq)
    # print("dK:", flash_dk)
    # print("dV:", flash_dv)

    dq_are_close = torch.allclose(dq, flash_dq, atol=1e-2)  # `atol` is the absolute tolerance
    dk_are_close = torch.allclose(dk, flash_dk, atol=1e-2)  # `atol` is the absolute tolerance
    dv_are_close = torch.allclose(dv, flash_dv, atol=1e-2)  # `atol` is the absolute tolerance

    assert dq_are_close, "Calculation of Q_grad is wrong"
    assert dk_are_close, "Calculation of K_grad is wrong"
    assert dv_are_close, "Calculation of V_grad is wrong"

if __name__ == "__main__":
    test_forward()
    test_backward()
