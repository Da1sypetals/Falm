# Flash Attention Lite
## Prerequisites:
- CUDA/CUDAtoolkit
- Torch 

## Run 
- Compile cpp-extension   
    ```
    bash local_build.sh
    ```
- Test flash-attention 
    ```
    python3 main.py
    ```

## Details:
    - which part should be loaded in sram?
        Qi, Kj, Vj, Sij 
        Qi, Kj, Vj: need to be shared amoung threads 
        Sij: size is dynamic
    - which part should be loaded in register?
        https://github.com/Dao-AILab/flash-attention/issues/618 
