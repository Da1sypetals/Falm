#include <cuda.h>
#include <stdio.h>

#define M 1024 // sram size

__global__ void foward_kernel(float* Q, float* K, float* V, float* O, float* l, float* m, int d) {
  // Q, K, V: query, key, value (N * d)
  // O, output: (N * d)
  // l, m: intermediate states (d) 
  // d, dimention (scaler, int)
  int thread_id = threadIdx.x;
  int batch_id, head_id = blockIdx.x, blockIdx.y;

  int Bc, Br = M / d, min(M/d, d); // block size  
  int Tc, Tr = N/br, N/br; // tile index 

  // Shared memory stored K, V, Q, O;
  extern shared_memory sram[];
  float* smem_K = sram;
  float* smem_V = &sram[Bc];
  float* smem_Q = &sram[Bc + Br];
  float* smem_O = &sram[Bc + Br * 2];

  for (int j = 0; j < Tc; ++j) {
    // Collaboratively Load Ki, Vj to the shared_memory ()
    for (int i = 0; i < Tr; ++i) {
      // Collaboratively Load Qi, Oi to shared memory 


      // Load l, m from HBM to register 

      l



    }
  } 
}

int main () {
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("%d", max_sram_size);
  return 0;
}
