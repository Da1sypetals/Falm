# Flash attention with lazily materialized attention mask

> This repo is a fork of https://github.com/weiyu0824/flash-attention-lite.


- Attention mask is constructed as $mask=ispositive(MM^T)$, where $M$ has shape `(seq_len, mask_dim)`. 
  - The $ispositive$ is an element-wise operation returning True if the element is positive, otherwise False.
  - `mask_dim` is often the same order of magnitude as `embed_dim` for $Q, K$ and $V$.
- `nan` can apppear when an entire row in attention mask is all zero.