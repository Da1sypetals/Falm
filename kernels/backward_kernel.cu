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
  
    for (int j = 0; j < Tc; ++j) {
      // Load Kj, Vj

      // Initialize dKj, dVj 

      for (int i = 0; i < Tr; ++i) {
        // Load Qi, Oi, dOi

        // Load li, mi

        // Compute Sij

        // Compute Pij 

        // Update dVj

        // Compute dPij 
        
        // Compute dSij 

        // Write updated dQi to HBM 
        
        // Update dKj 
        
      }
      // Write dKj, dVj 
    }

}
