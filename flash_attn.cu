#include <cuda.h>
#include <stdio.h>

#define M 1024  // sram size

__global__ void foward_kernel(float *Q, float *K, float *V, float *O, float *l,
                              float *m, int N, int d, int Bc, int Br) {
    // Q, K, V: query, key, value (N * d)
    // O, output: (N * d)
    // l, m: intermediate states (N)
    // N: sequence length (scaler, int)
    // d: dimention (scaler, int)
    // Bc, Br: col/row block size
    int thread_id = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int batch_size = gridDim.x;
    int num_head = gridDim.y;

    // Differnt offset for different (batch, head)
    int qkv_offset = (batch_id * num_head * N * d) + (head_id * N * d);
    int lm_offset = (batch_id * num_head * N) + (head_id * N);

    // Number of tile
    int Tc = ceil(N / Bc);
    int Tr = ceil(N / Br);

    // Shared memory stored K, V, Q, O;
    extern __shared__ float sram[];
    float *smem_Kj = sram;
    float *smem_Vj = &sram[Bc];
    float *smem_Qi = &sram[Bc + Br];
    float *smem_Oi = &sram[Bc + Br * 2];

    for (int j = 0; j < Tc; ++j) {
        // Collaboratively Load Ki, Vj to the shared_memory ()
        // Kj, Vj: Bc*d
        for (int y = thread_id; y < Bc; y += Br) {
            for (int x = 0; 0 < d; x += 1) {
                // TODO: do we need to check range here
                smem_Kj[y * d + x] = K[qkv_offset + (j * Bc * d) + y * d + x];
                smem_Vj[y * d + x] = V[qkv_offset + (j * Bc * d) + y * d + x];
            }
        }
        __syncthreads();

        for (int i = 0; i < Tr; ++i) {
            // Collaboratively Load Qi, Oi to shared memory
            for (int x = 0; x < d; x += 1) {
                smem_Qi[thread_id * d + x] = Q[qkv_offset + thread_id * d + x];
                smem_Oi[thread_id * d + x] = O[qkv_offset + thread_id * d + x];
            }
            __syncthreads();

            // Load li, mi from HBM to register
            float li = l[lm_offset + i + thread_id];
            float mi = m[lm_offset + i + thread_id];

            // Compute Sij = Qi * Kj^transpose
            float Si[Bc];
            for (int c = 0; c < Bc; c += 1) {
                Si[c] = 0;
                for (int x = 0; x < d; x += 1) {
                    Si[c] += smem_Qi[thread_id * d + x] * smem_Kj[c * d + x];
                }
            }
            // Find new maximum mi
            float mi_tilde = Si[0];  // maximum inside this block
            for (int c = 1; c < Bc; c += 1) {
                mi_tilde = max(mi_tilde, Si[c]);
            }

            // Calculate Pi
            float Pi[Bc];  // Bc
            float li_tilde = 0;
            for (int c = 1; c < Bc; c += 1) {
                Pi[c] = __expf(Si[c] - mi_tilde);
                li += Pi[c];
            }

            // Compute mi, li
            float mi_new = max(mi_tilde, mi);
            float li_new =
                __expf(mi - mi_new) * li + __expf(mi_tilde - mi_new) * li_tilde;

            // Write Oi to HBM
            for (int x = 0; x < d; x += 1) {
                // Calculate Pij * Vj
                float pv = 0;
                for (int c = 0; c < Bc; c += 1) {
                    pv += Pi[c] * smem_Vj[c * d + x];
                }

                O[qkv_offset + (i * Br * d) + (thread_id * d) + x] =
                    (1 / li_new) *
                    ((li * __expf(mi - mi_tilde) * smem_Oi[thread_id * d + x]) +
                     __expf(mi_tilde - mi_new * pv));
            }

            // Write li, mi to HBM
            l[lm_offset + i + thread_id] = li_new;
            m[lm_offset + i + thread_id] = mi_new;

            // Make sure K, V cache could be correct.
            __syncthreads();
        }
    }
}

// torch::Tensor forward(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V) {
//
// }

int main() {
    int max_sram_size;

    int Bc, Br = M / d, min(M / d, d);  // block size
    // int Bc, Br = 64, 64;
    // #threads = Br
    int Tc, Tr = N / br, N / br;  // tile index

    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock,
                           0);
    printf("%d", max_sram_size);

    dim3 grid_dim(batch_size, num);
    dim3 block_dim();

    return 0;
}
