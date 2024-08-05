# Flash Attention Lite

## Details:
    - which part should be loaded in sram?
        Qi, Kj, Vj, Sij, Oi 
        Qi, Kj, Vj, Oi on shared memory
        Sij on register. (Do not need to shared across threads)
    - which part should be loaded in register?
        https://github.com/Dao-AILab/flash-attention/issues/618 
