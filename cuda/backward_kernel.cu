#include <cuda.h>
#include <torch/extension.h>


__global__ void backward_kernel(float *Q, float *K, float *V, float *O, float *dQ, float *dK, float *dV, float *dO,
                                float *l, float *m, const int N, const int d, const int Tc, const int Tr, const float scale,
                                const int32_t* mf, const int dm) {
    // Given Q, K, V, O, dO, l, m, we need to compute dQ, dK, dV
    //
    // Q, K, V, O: query, key, value, output (N * d)
    // l, m: intermediate states (N)
    // N: sequence length <int>(scaler)
    // d: dimention <int>(scaler)
    // Tc, Tr: number of blocks

    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int num_heads = gridDim.y;
    int block_size = blockDim.x;  // tile_size

    // Differnt offset for different (batch, head)
    int qkvo_offset = (batch_id * num_heads * N * d) + (head_id * N * d);
    int mask_offset = batch_id * N * dm;
    int lm_offset = (batch_id * num_heads * N) + (head_id * N);

    extern __shared__ float sram[];
    // K, V, Q, O, dK, dV, dO has size=block_size * d
    // SPij, dSPij has size=block_size * block_size
    float *const smem_Kj = &sram[0];
    float *const smem_Vj = &sram[block_size * d];
    float *const smem_Oi = &sram[block_size * d * 2];
    float *const smem_Qi = &sram[block_size * d * 3];
    float *const smem_dKj = &sram[block_size * d * 4];
    float *const smem_dVj = &sram[block_size * d * 5];
    float *const smem_dOi = &sram[block_size * d * 6];
    float *const smem_SPij = &sram[block_size * d * 7];
    float *const smem_dSPij = &sram[block_size * d * 7 + block_size * block_size];
    int32_t *const smem_Mi = reinterpret_cast<int32_t *const>(&sram[block_size * d * 7 + block_size * block_size * 2]);
    int32_t *const smem_Mj = reinterpret_cast<int32_t *const>(&sram[block_size * d * 7 + block_size * block_size * 2 + block_size * dm]);

    int offset_si = block_size * thread_id;

    for (int j = 0; j < Tc; ++j) {
        const int num_cols = min(block_size, N - (block_size * j));
        const int global_col = j * block_size + thread_id;

        // Load Kj, Vj to shared memory
        if (global_col < N) {  // Make sure global col < seq_len
            for (int x = 0; x < d; x += 1) {
                smem_Kj[thread_id * d + x] = K[qkvo_offset + (j * block_size * d) + (thread_id * d) + x];
                smem_Vj[thread_id * d + x] = V[qkvo_offset + (j * block_size * d) + (thread_id * d) + x];
            }

            // load Mj to sram
            for(int x = 0; x < dm; x++) {
                smem_Mj[thread_id * dm + x] = mf[mask_offset + (j * block_size * dm) + (thread_id * dm) + x];
            }
        }
        // Init dKj, dVj on shared_memory
        for (int x = 0; x < d; x += 1) {
            smem_dKj[thread_id * d + x] = 0;
            smem_dVj[thread_id * d + x] = 0;
        }
        
        __syncthreads();

        for (int i = 0; i < Tr; ++i) {
            const int num_rows = min(block_size, N - (block_size * i));
            const int global_row = i * block_size + thread_id; 
   
            if (global_row < N) {  // Make sure global row < seq_len
                // Load Qi, dOi to register
                for (int x = 0; x < d; x += 1) {
                    smem_Qi[thread_id * d + x] = Q[qkvo_offset + (i * block_size * d) + (thread_id * d) + x];
                    smem_Oi[thread_id * d + x] = O[qkvo_offset + (i * block_size * d) + (thread_id * d) + x];
                    smem_dOi[thread_id * d + x] = dO[qkvo_offset + (i * block_size * d) + (thread_id * d) + x];
                }

                for (int x = 0; x < dm; x++) {
                    smem_Mi[thread_id * dm + x] = mf[mask_offset + (i * block_size * dm) + (thread_id * dm) + x];
                }

                __syncthreads();

                // Load li, mi from HBM to register
                float li = l[lm_offset + (i * block_size) + thread_id];
                float mi = m[lm_offset + (i * block_size) + thread_id];

                // Compute Sij
                // Sij = Qi * Kj
                for (int c = 0; c < num_cols; c += 1) {
                    float dot = 0;
                    for (int x = 0; x < d; x += 1) {
                        dot += (smem_Qi[thread_id * d + x] * smem_Kj[c * d + x]);
                    }
                    smem_SPij[offset_si + c] = dot * scale;
                }

                // M = Mi @ Mj^T, apply attention bias
                for (int c = 0; c < num_cols; c += 1) {
                    int masksum = 0;
                    for (int x = 0; x < dm; x += 1) {
                        masksum += (smem_Mi[thread_id * dm + x] * smem_Mj[c * dm + x]);
                    }
                    smem_SPij[offset_si + c] += masksum > 0 ? 0.0 : -INFINITY;
                }

                // Compute Pij
                // Pij = 1/li * Sij
                for (int c = 0; c < num_cols; c += 1) {
                    smem_SPij[offset_si + c] = __expf(smem_SPij[offset_si + c] - mi) / li;
                }
            }
            __syncthreads();

            
            if (global_col < N) {
                // Note: Check global column because each thread should write 1 column to dVj
                // Update dVj
                // dVj += Pij_transpose * dOi
                for (int x = 0; x < d; x += 1) {
                    float dot = 0;
                    for (int r = 0; r < num_rows; r += 1) {
                        dot += smem_SPij[r * block_size + thread_id] * smem_dOi[r * d + x];
                    }
                    smem_dVj[thread_id * d + x] += dot;
                }
            }

            if (global_row < N) {
                // Compute dPij
                // dPij = dOi * Vj_transpose
                for (int c = 0; c < num_cols; c += 1) {
                    float dot = 0;
                    for (int x = 0; x < d; x += 1) {
                        dot += smem_dOi[thread_id * d + x] * smem_Vj[c * d + x];
                    }
                    smem_dSPij[thread_id * block_size + c] = dot;
                }

                // Compute dSij
                // Di = row_sum(dOi o Oi)
                // dSij = Pij o (dPij - Di)
                float Di = 0;
                for (int x = 0; x < d; x += 1) {
                    Di += smem_dOi[thread_id * d + x] * smem_Oi[thread_id * d + x];
                }
                for (int c = 0; c < num_cols; c += 1) {
                    smem_dSPij[thread_id * block_size + c] =
                        smem_SPij[thread_id * block_size + c] * (smem_dSPij[thread_id * block_size + c] - Di);
                }
                __syncthreads();

                // Write updated dQi to HBM
                // dQi += dSij * Kj
                for (int x = 0; x < d; x += 1) {
                    float dot = 0;
                    for (int c = 0; c < num_cols; c += 1) {
                        dot += smem_dSPij[thread_id * block_size + c] * smem_Kj[c * d + x];
                    }
                    dQ[qkvo_offset + (i * block_size * d) + (thread_id * d) + x] += dot * scale;
                }
            }
            
            if (global_col < N) {
                // Update dKj
                // dKj += dSij_transpose * Qi
                for (int x = 0; x < d; x += 1) {
                    float dot = 0;
                    for (int r = 0; r < num_rows; r += 1) {
                        dot += smem_dSPij[r * block_size + thread_id] * smem_Qi[r * d + x];
                    }
                    smem_dKj[thread_id * d + x] += dot * scale;
                }
            }
            // Make sure Qi, O, dOi load correctly
            __syncthreads();
        }

        // Write dKj, dVj to HBM
        if ((j * block_size + thread_id) < N) { 
            for (int x = 0; x < d; x += 1) {
                dK[qkvo_offset + (j * block_size * d) + (thread_id * d) + x] = smem_dKj[thread_id * d + x];
                dV[qkvo_offset + (j * block_size * d) + (thread_id * d) + x] = smem_dVj[thread_id * d + x];
            }
        }
        __syncthreads();
    }
}

inline void CHECK_CUDA_ERROR() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

void launch_backward_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor dQ,
                            torch::Tensor dK, torch::Tensor dV, torch::Tensor dO, torch::Tensor l, torch::Tensor m, torch::Tensor mf) {
    //
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);
    int dm = mf.size(2);

    printf("Batch=%d, Head=%d, SeqLen=%d, EmbDim=%d\n", batch_size, num_heads, N, d);
    //
    int max_shared_memory;
    int max_threads_num;
    cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&max_threads_num, cudaDevAttrMaxThreadsPerBlock, 0);

    // Fix Tile size
    const int block_size = 32;
    int Bc = block_size;
    int Br = block_size;
    int Tc = (int)std::ceil((float)N / Bc);
    int Tr = (int)std::ceil((float)N / Br);

    dim3 grid_dim(batch_size, num_heads);
    dim3 thread_block_dim(block_size);
    // shared_memory_size
    // For: Kj, Vj, Qi, Oi, dKj, dVj, dOi, SPij, dSPij, Mi, Mj
    const int shared_memory_size = sizeof(float) * ((7 * block_size * d) + (2 * block_size * block_size)) + sizeof(int32_t) * (2 * block_size * dm);

    printf("Max_shared(bytes)=%d, Requested_memory(bytes)=%d\n", max_shared_memory, shared_memory_size);
    TORCH_CHECK(shared_memory_size < max_shared_memory, "Shared memory size exceeds the device limit");

    printf("N=%d, d=%d, block_size=%d, Tc=%d, Tr=%d\n", N, d, block_size, Tc, Tr);

    float scale = 1.0f / std::sqrt(static_cast<float>(K.size(3)));

    // Launch
    backward_kernel<<<grid_dim, thread_block_dim, shared_memory_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), dQ.data_ptr<float>(),
        dK.data_ptr<float>(), dV.data_ptr<float>(), dO.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), N,
        d, Tc, Tr, scale, mf.data_ptr<int32_t>(), dm);

    // CHECK_CUDA_ERROR();
}
