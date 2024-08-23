# Flash Attention Lite
This repository provides a very basic implementation of FlashAttention-1, including both the forward and backward passes. The goal of this project is to deepen my understanding of the FlashAttention mechanism.

## Prerequisites:
- CUDA/CUDAtoolkit
- Torch 

## Run Code 
- Check out `kernels/` directory to review the minimal implementaion.
    
- To compile cpp-extension, run: 
    ```
    bash local_build.sh
    ```
- To test flash-attention, run:
    ```
    python3 main.py
    ```
    
## Details
- Forward pass: Implements follow dynamic block size described in the FlashAttention paper. The block size for rows and columns are different, with the number of threads per block equal to the block size of the rows.
- Backward pass: Uses a fixed block size of 32 for both rows and columns, with the number of threads per block also set to 32 for simplicity. The author metioned this simplification in [this issue](https://github.com/Dao-AILab/flash-attention/issues/618). 
- Data Type: All tensors are fixed to the float data type for simplicity.

## Reference
- https://github.com/tspeterkim/flash-attention-minimal
