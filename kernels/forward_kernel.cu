#include <cmath>
#include "flash_attn_kernel.h"

__global__ void forward_kernel(float *Q, float *K, float *V, float *O, float *l,
                              float *m, const int N, const int d, const int Bc, const int Br, const int Tc, const int Tr) {
    // Given Q, K, V, we need to compute O 
    //
    // Q, K, V: query, key, value (N * d)
    // O, output: (N * d)
    // l, m: intermediate states (N)
    // N: sequence length <int>(scaler)
    // d: dimention <int>(scaler)
    // Bc, Br: number of col/row per block
    // Tc, Tr: number of blocks 
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int num_head = gridDim.y;
    int num_threads = blockDim.x; //num_threads=Br


    // Differnt offset for different (batch, head)
    int qkv_offset = (batch_id * num_head * N * d) + (head_id * N * d);
    int lm_offset = (batch_id * num_head * N) + (head_id * N);

    // Shared memory stored K, V, Q, SP;
    // Note: SP stored Sij and Pij
    // Note: Why SPij in shared memory? because their size is dynamic
    extern __shared__ float sram[];

    float * const smem_Kj = &sram[0]; // size=(Bc * d) 
    float * const smem_Vj = &sram[Bc * d]; // size=(Bc * d)
    float * const smem_Qi = &sram[Bc * d + Bc * d]; // size=(Br * d)
    float * const smem_SPij = &sram[Bc * d + Bc * d + Br * d]; // size=(Bc * Br)

    // printf("thread=%d shared: %d, %d, %d\n", thread_id, Vj_offset, Qi_offset, SPij_offset);
    const int offset_si = thread_id * Bc; // Different thread process different row of Sij 

    for (int j = 0; j < Tc; ++j) {
        // Collaboratively Load Ki, Vj to the shared_memory ()
        // Kj, Vj: Bc*d
        for (int y = thread_id; y < Bc; y += num_threads) {
            if ((j * Bc + y) < N) { // Make sure global col < seq_len 
            
              for (int x = 0; x < d; x += 1) {
                  smem_Kj[y * d + x] = K[qkv_offset + (j * Bc * d) + (y * d) + x];
                  smem_Vj[y * d + x] = V[qkv_offset + (j * Bc * d) + (y * d) + x];

                  // printf("thread=%d, load q=%f\n", thread_id, smem_Kj[offset_si + c]);
                  // printf("thread=%d, load k=%f\n", thread_id, smem_Vj[offset_si + c]);
              }
            }
        }
        __syncthreads();

        const int num_cols = min(Bc, N - (Bc * j));
        for (int i = 0; i < Tr; ++i) {
            if ((i * Br + thread_id) < N){ // Make sure global row < seq_len

            // Collaboratively Load Qi to shared memory
            for (int x = 0; x < d; x += 1) {
              smem_Qi[thread_id * d + x] = Q[qkv_offset + (i * Br * d) + (thread_id * d) + x];
              // if (thread_id == 2){
              //   printf("thread=%d, load q=%f\n", thread_id, smem_Qi[thread_id * d + x]);
              // }
            }
            __syncthreads();

            // Load li, mi from HBM to register
            float li = l[lm_offset + (i * Br) + thread_id];
            float mi = m[lm_offset + (i * Br) + thread_id];

            // Compute Sij = Qi * Kj^transpose
            for (int c = 0; c < num_cols; c += 1) {
                // smem_SPij[offset_si + c] = 0;
                // printf("rewrite offset=%d\n", offset_si + c);
                float row_sum = 0;
                for (int x = 0; x < d; x += 1) {
                    row_sum += (smem_Qi[thread_id * d + x] * smem_Kj[c * d + x]);
                    // printf("thread=%d, (c, x)=(%d, %d), q-read from sram[%d], compute q*k=%f*%f\n", 
                    //     thread_id, c, x, (2 * Bc * d) + (thread_id * d + x), smem_Qi[thread_id * d + x], smem_Kj[c * d + x]);
                }
                // printf("thread=%d, writes: %f to sram[%d] Sij_off=%d + %d (Bc=%d, Br=%d, d=%d)\n", 
                //     thread_id, row_sum, (Bc * d + Bc * d + Br * d) + offset_si + c, (Bc * d + Bc * d + Br * d), offset_si + c, Bc, Br, d); 
                smem_SPij[offset_si + c] = row_sum;
            }

            // Find new maximum mi for each row
            float mi_tilde = -INFINITY;  // maximum inside this block
            for (int c = 0; c < num_cols; c += 1) {
                mi_tilde = max(mi_tilde, smem_SPij[offset_si + c]);
                // printf("thread=%d, spij=%f\n", thread_id, s mem_SPij[offset_si + c]);
            }

            // Calculate Pij & li_tilde
            float li_tilde = 0;
            for (int c = 0; c < num_cols; c += 1) {
                smem_SPij[offset_si + c] = __expf(smem_SPij[offset_si + c] - mi_tilde);
                li_tilde += smem_SPij[offset_si + c];
            }

            // Compute mi_new, li_new
            float mi_new = max(mi, mi_tilde);
            float li_new = __expf(mi - mi_new) * li + __expf(mi_tilde - mi_new) * li_tilde;


            // printf("thread=%d, num_cols=%d, mi=%f, mi_tilde=%f, mi_new=%f, li=%f, li_tilde=%f, li_new=%f\n", thread_id, num_cols, mi, mi_tilde, mi_new, li, li_tilde, li_new);
            // Write Oi to HBM
            for (int x = 0; x < d; x += 1) {
                // Calculate Pij * Vj
                float pv = 0;
                for (int c = 0; c < num_cols; c += 1) {
                    pv += smem_SPij[offset_si + c] * smem_Vj[c * d + x];
                }

                O[qkv_offset + (i * Br * d) + (thread_id * d) + x] =
                    (1 / li_new) *
                    ((li * __expf(mi - mi_new) * O[qkv_offset + (i * Br * d) + (thread_id * d) + x]) +
                     __expf(mi_tilde - mi_new) * pv);
            }

            // Write li, mi to HBM
            l[lm_offset + (i * Br) + thread_id] = li_new;
            m[lm_offset + (i * Br) + thread_id] = mi_new;
            } 
            // Make sure K, V cache could be correct.
            __syncthreads();
        }
    }
}

inline void CHECK_CUDA_ERROR() {                                          
    cudaError_t err = cudaGetLastError();                            
    if (err != cudaSuccess) {                                         
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; 
        exit(err);                                                    
    }                                                                 
}

void launch_forward_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor l, torch::Tensor m, torch::Tensor O) {
    // 
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);
    printf("Batch=%d, Head=%d, SeqLen=%d, EmbDim=%d\n", batch_size, num_heads, N, d);
    //
    int max_shared_memory;
    int max_threads_num;
    cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&max_threads_num, cudaDevAttrMaxThreadsPerBlock, 0);
    //TODO: Dynamic datatype 
    int M = max_shared_memory / sizeof(float); // number of floats shared memory can hold
    int Bc = std::ceil(M / (4 * d));        
    int Br = std::min(std::min(Bc, d), max_threads_num);
    int Tc = (int)std::ceil(float(N) / Bc);
    int Tr = (int)std::ceil(float(N) / Br);

    dim3 grid_dim(batch_size, num_heads);
    dim3 thread_block_dim(Br);
    // shared_memory_size
    // For: Ki, Vi, Qi, Sij 
    const int shared_memory_size = sizeof(float) * ((2 * Bc * d) + (Br * d) + (Bc * Br));
    
    printf("Max_shared(bytes)=%d, Max_shared(#dtype)=%d, Requested_memory(bytes)=%d\n", max_shared_memory, M, shared_memory_size);
    TORCH_CHECK(shared_memory_size < max_shared_memory, "Shared memory size exceeds the device limit"); 
    
    printf("N=%d, d=%d, Bc=%d, Br=%d, Tc=%d, Tr=%d\n", N, d, Bc, Br, Tc, Tr);
    printf("Start Position: K=0, V=%d, Q=%d, S=%d\n", Bc * d, 2 * Bc * d, (2 * Bc * d) + (Br * d));
    // Launch
    forward_kernel<<<grid_dim, thread_block_dim, shared_memory_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
        O.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), 
        N, d, Bc, Br, Tc, Tr
    );
    
    CHECK_CUDA_ERROR();
}

