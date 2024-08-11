#include <cuda.h>
#include "flash_attn_kernel.h"


__global__ void backward_kernel(float *Q, float *K, float *V, float *O, float *l,
                              float *m, const int N, const int d, const int Bc, const int Br, const int Tc, const int Tr) {
    
    // Given Q, K, V, O, dO, l, m we need to compute dQ, dK, dV
    // 
    // Q, K, V: query, key, value (N * d)
    // O, output: (N * d)
    // l, m: intermediate states (N)

    // N: sequence length <int>(scaler)
    // d: dimention <int>(scaler)
    // Bc, Br: number of col/row per block
    // Tc, Tr: number of blocks 
 
    extern float* sram;
    float * const smem_Kj;
    float * const smem_Vj;
    float * const 

    for (int j = 0; j < Tc; ++j) {
      // Load Kj, Vj to shared memory


      // Initialize dKj, dVj on register

      for (int i = 0; i < Tr; ++i) {
        // Load Qi, dOi to shared_memory 

        // Load Oi to register 

        // Load li, mi to register

        // Compute Sij
        // Sij = Qi * Kj 

        // Compute Pij 
        // Pij = 1/li * Sij

        // Update dVj
        dVj += Pij * 

        // Compute dPij 
        
        // Compute dSij 

        // Write updated dQi to HBM 
        
        // Update dKj 
        
      }
      // Write dKj, dVj 
    }

}

inline void CHECK_CUDA_ERROR() {                                          
    cudaError_t err = cudaGetLastError();                            
    if (err != cudaSuccess) {                                         
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; 
        exit(err);                                                    
    }                                                                 
}

void lanuch_backward_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor l, torch::Tensor m, torch::Tensor O) {
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
    backward_kernel<<<grid_dim, thread_block_dim, shared_memory_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
        O.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), 
        N, d, Bc, Br, Tc, Tr
    );
    
    CHECK_CUDA_ERROR();
}

