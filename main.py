import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

# Load the CUDA kernel as a python module
flash_attn = load(
    name="flash_attn",
    sources=[
        "cuda/main.cpp",
        "cuda/forward_kernel.cu",
        "cuda/backward_kernel.cu",
    ],
    extra_cuda_cflags=["-O2"],
)

torch.manual_seed(0)


def test_forward():
    batch_size = 10
    n_head = 4
    seq_len = 4096
    head_embd = 16

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    print("======Attention======")
    o = F.scaled_dot_product_attention(q, k, v)
    # print(o)

    print("======Flash Attention======")
    flash_o, flash_l, flash_m = flash_attn.forward(q, k, v)
    # print(flash_o)
    # print(flash_l)
    # print(flash_m)

    are_close = torch.allclose(
        o, flash_o, atol=1e-2
    )  # `atol` is the absolute tolerance
    assert are_close, "foward is wrong"

    diff = o - flash_o
    meandiff = diff.abs().mean() / flash_o.abs().mean()
    print(f"Mean diff = {meandiff}")


def test_backward():
    batch_size = 5
    n_head = 2
    seq_len = 1024
    head_embed = 16

    q = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()

    print("======Attention======")
    s = torch.matmul(q, k.transpose(-2, -1))
    scale = 1 / (k.shape[3] ** 0.5)
    s = s * scale
    m = torch.max(s, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(s - m)
    l = torch.sum(exp_scores, dim=-1, keepdim=True)
    p = exp_scores / l
    o = torch.matmul(p, v)

    # random init o's gradient
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

    print("======Flash Attention======")
    flash_dq, flash_dk, flash_dv = flash_attn.backward(q, k, v, o, do, l, m)
    # print("dQ:", flash_dq)
    # print("dK:", flash_dk)
    # print("dV:", flash_dv)

    dq_are_close = torch.allclose(
        dq, flash_dq, atol=1e-2
    )  # `atol` is the absolute tolerance
    dk_are_close = torch.allclose(
        dk, flash_dk, atol=1e-2
    )  # `atol` is the absolute tolerance
    dv_are_close = torch.allclose(
        dv, flash_dv, atol=1e-2
    )  # `atol` is the absolute tolerance

    qdiff = dq - flash_dq
    kdiff = dk - flash_dk
    vdiff = dv - flash_dv
    print(f"Q diff: {(qdiff.abs().mean()) / q.abs().mean()}")
    print(f"K diff: {(kdiff.abs().mean()) / k.abs().mean()}")
    print(f"V diff: {(vdiff.abs().mean()) / v.abs().mean()}")

    assert dq_are_close, "Calculation of Q_grad is wrong"
    assert dk_are_close, "Calculation of K_grad is wrong"
    assert dv_are_close, "Calculation of V_grad is wrong"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", action="store_true")
    parser.add_argument("-b", action="store_true")
    args = parser.parse_args()

    if args.f:
        test_forward()
    if args.b:
        test_backward()

    if not args.f and not args.b:
        print("No test is running, quitting...")
