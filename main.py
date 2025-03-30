import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F
import einops as ein

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


def make_mask(batch_size, seq_len, mask_embed, n_head):
    print("Make mask start")
    i = 0
    while True:
        i += 1
        mf = (torch.rand(batch_size, seq_len, mask_embed) < 0.065).to(dtype=torch.int32)
        # mf = torch.ones(batch_size, seq_len, seq_len, dtype=torch.int32)
        mask = (ein.einsum(mf, mf, "b l1 d, b l2 d -> b l1 l2") > 0).to(torch.bool)
        mf = mf.cuda()
        mask = ein.repeat(mask, "b l1 l2 -> b h l1 l2", h=n_head).cuda()

        if (mask.sum(dim=-1) > 0.1).all():
            print(f"Make mask ok at iteration {i}")
            break
        else:
            print(f"Make mask failed at iteration {i}, retrying...")
            print

    return mf, mask


def test_forward():
    batch_size = 1
    n_head = 4
    seq_len = 64
    head_embd = 16
    mask_embed = seq_len

    mf, mask = make_mask(batch_size, seq_len, mask_embed, n_head)

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    print("======Attention======")
    o = F.scaled_dot_product_attention(q, k, v, mask)
    # print(o)

    print("======Flash Attention======")
    flash_o, flash_l, flash_m = flash_attn.forward(q, k, v, mf)
    # print(flash_o)
    # print(flash_l)
    # print(flash_m)

    diff = o - flash_o
    meandiff = diff.abs().mean() / o.abs().mean()
    print(f"Mean diff = {meandiff}")

    o_no_mask = F.scaled_dot_product_attention(q, k, v)
    diffno = o_no_mask - flash_o
    meandiff = diffno.abs().mean() / o_no_mask.abs().mean()
    print(f"Mean diff with no mask = {meandiff}")


def test_backward():
    batch_size = 5
    n_head = 4
    seq_len = 2048
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
