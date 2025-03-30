import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F
import einops as ein
from torch.autograd import Function

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


class FlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, mf):
        # Save inputs for backward pass

        # Call the CUDA forward function
        output, l, m = flash_attn.forward(q, k, v, mf)
        ctx.save_for_backward(q, k, v, l, m, mf, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, l, m, mf, output = ctx.saved_tensors

        # Call the CUDA backward function
        dq, dk, dv = flash_attn.backward(q, k, v, output, grad_output, l, m, mf)

        return dq, dk, dv, None


# Convenience function that uses the autograd Function
def flash_attention(q, k, v, mf):
    """
    PyTorch function for simplified Flash Attention.

    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        mf: additional parameter for FlashAttention

    Returns:
        output tensor
    """
    return FlashAttentionFunction.apply(q, k, v, mf)


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
    batch_size = 1
    n_head = 4
    seq_len = 64
    head_embed = 16
    mask_embed = seq_len

    mf, mask = make_mask(batch_size, seq_len, mask_embed, n_head)

    # Original tensors with requires_grad=True
    q = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embed, requires_grad=True).cuda()

    print("======Attention======")
    # o = F.scaled_dot_product_attention(q, k, v, mask)
    o = F.scaled_dot_product_attention(q, k, v)
    res = o.sum()

    # Retain gradients before backward pass
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    res.backward()

    dq = q.grad
    dk = k.grad
    dv = v.grad

    print("======Flash Attention======")
    # Clone tensors but keep them in the computation graph
    fq = q.clone().requires_grad_(True)
    fk = k.clone().requires_grad_(True)
    fv = v.clone().requires_grad_(True)

    o_flash = FlashAttentionFunction.apply(fq, fk, fv, mf)
    res_flash = o_flash.sum()

    # Retain gradients before backward pass
    fq.retain_grad()
    fk.retain_grad()
    fv.retain_grad()

    res_flash.backward()

    flash_dq = fq.grad
    flash_dk = fk.grad
    flash_dv = fv.grad

    print(f"dQ max: {flash_dq.abs().max()}")
    print(f"dK max: {flash_dk.abs().max()}")
    print(f"dV max: {flash_dv.abs().max()}")

    qdiff = dq - flash_dq
    kdiff = dk - flash_dk
    vdiff = dv - flash_dv
    print(f"Q diff: {(qdiff.abs().mean()) / q.abs().mean()}")
    print(f"K diff: {(kdiff.abs().mean()) / k.abs().mean()}")
    print(f"V diff: {(vdiff.abs().mean()) / v.abs().mean()}")

    dq_are_close = torch.allclose(dq, flash_dq, atol=1e-2)
    dk_are_close = torch.allclose(dk, flash_dk, atol=1e-2)
    dv_are_close = torch.allclose(dv, flash_dv, atol=1e-2)

    # assert dq_are_close, "Calculation of Q_grad is wrong"
    # assert dk_are_close, "Calculation of K_grad is wrong"
    # assert dv_are_close, "Calculation of V_grad is wrong"


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
