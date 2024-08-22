#include <cmath>
#include <algorithm>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "ATen/core/TensorBody.h"
#include "c10/util/Exception.h"
#include "flash_attn_kernel.h"
#include <tuple>

// Forward function declaration
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Implementation of the forward pass (CUDA)

    // Ensure tensors are on the same device
    auto device = Q.device();
    TORCH_CHECK(K.device() == device, "K must be on the same device as Q");
    TORCH_CHECK(V.device() == device, "V must be on the same device as Q");
    
    // Size checking 
    auto num_dim = Q.dim();
    TORCH_CHECK(num_dim == 4, "Q must have 4 dimension (batch_size, num_head, seq_len, embed_dim)");
    auto sizes = Q.sizes();
    TORCH_CHECK(K.sizes() == sizes, "K must have same size as Q");
    TORCH_CHECK(V.sizes() == sizes, "V must have same size as Q");

    // Prepare output tensor 
    int batch_size = sizes[0];
    int num_heads = sizes[1]; 
    int N = sizes[2];
    int d = sizes[3];
    
    // l, s, O
    torch::Tensor l = torch::zeros({batch_size, num_heads, N}).cuda();
    torch::Tensor m = torch::full({batch_size, num_heads, N}, -INFINITY).cuda();    
    torch::Tensor O = torch::zeros({batch_size, num_heads, N, d}).cuda();
    
    // std::string ls = l.toString();
    // std::string ms = m.toString();
    // std::string os = O.toString();

    // std::cout << l << std::endl;
    // std::cout << m << std::endl;
    // std::cout << O << std::endl;

    lanuch_forward_kernel(Q, K, V, l, m, O);
    return O;
}

// Backward function declaration
torch::Tensor backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor dO, torch::Tensor l, torch::Tensor m) {
    // Implementation of the backward pass
    // Call CUDA kernels or perform computations here
    // Ensure tensors are on the same device
    auto device = Q.device();
    TORCH_CHECK(K.device() == device, "K must be on the same device as Q");
    TORCH_CHECK(V.device() == device, "V must be on the same device as Q");
    TORCH_CHECK(O.device() == device, "O must be on the same device as Q");
    TORCH_CHECK(dO.device() == device, "dO must be on the same device as Q");
    TORCH_CHECK(l.device() == device, "l must be on the same device as Q");
    TORCH_CHECK(m.device() == device, "m must be on the same device as Q");
    
    // Size checking 
    auto num_dim = Q.dim();
    TORCH_CHECK(num_dim == 4, "Q must have 4 dimension (batch_size, num_head, seq_len, embed_dim)");
    auto sizes = Q.sizes();
    TORCH_CHECK(K.sizes() == sizes, "K must have same size as Q");
    TORCH_CHECK(V.sizes() == sizes, "V must have same size as Q");
    TORCH_CHECK(O.sizes() == sizes, "O must have same size as Q");
    TORCH_CHECK(dO.sizes() == sizes, "dO must have same size as Q");
    
    // Prepare output tensor 
    int batch_size = sizes[0];
    int num_heads = sizes[1]; 
    int N = sizes[2];
    int d = sizes[3];

    TORCH_CHECK(l.sizes() == std::make_tuple(batch_size, num_heads, N), "l must have compatible size with QKVO");
    TORCH_CHECK(m.sizes() == std::make_tuple(batch_size, num_heads, N), "m must have compatible size with QKVO");

    //dQ, dK, dV  

    torch::Tensor dQ = torch::zeros({batch_size, num_heads, N, d}).cuda();
    torch::Tensor dK = torch::zeros({batch_size, num_heads, N, d}).cuda();
    torch::Tensor dV = torch::zeros({batch_size, num_heads, N, d}).cuda();

    return ;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define the forward function in the module
    m.def("forward", &forward, "Flash Attention forward (CUDA)");
    
    // Define the backward function in the module
    m.def("backward", &backward, "Flash Attention backward (CUDA)");
    
}
