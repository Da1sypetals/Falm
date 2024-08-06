#include <cmath>
#include <torch/extension.h>

// Forward function declaration
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Implementation of the forward pass
    // Call CUDA kernels or perform computations here
    // Ensure tensors are on the same device
    auto device = Q.device();
    TORCH_CHECK(K.device() == device, "K must be on the same device as Q");
    TORCH_CHECK(V.device() == device, "V must be on the same device as Q");

    // Compute attention scores (Q * K^T)
    auto scores = torch::matmul(Q, K.transpose(-2, -1));
    
    // Scale scores by sqrt(d_k)
    auto d_k = Q.size(-1);
    auto scaling_factor = std::sqrt(d_k);
    scores = scores / scaling_factor;

    // Apply softmax to get attention weights
    auto attention_weights = torch::softmax(scores, /*dim=*/-1);

    // Compute the weighted sum of values (attention_weights * V)
    auto output = torch::matmul(attention_weights, V);

    return output;
}

// Backward function declaration
torch::Tensor backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Implementation of the backward pass
    // Call CUDA kernels or perform computations here
    // Ensure tensors are on the same device
    auto device = Q.device();
    TORCH_CHECK(K.device() == device, "K must be on the same device as Q");
    TORCH_CHECK(V.device() == device, "V must be on the same device as Q");

    // Compute attention scores (Q * K^T)
    auto scores = torch::matmul(Q, K.transpose(-2, -1));
    
    // Scale scores by sqrt(d_k)
    auto d_k = Q.size(-1);
    auto scaling_factor = std::sqrt(d_k);
    scores = scores / scaling_factor;

    // Apply softmax to get attention weights
    auto attention_weights = torch::softmax(scores, /*dim=*/-1);

    // Compute the weighted sum of values (attention_weights * V)
    auto output = torch::matmul(attention_weights, V);

    return output;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define the forward function in the module
    m.def("forward", &forward, "Flash Attention forward (CUDA)");
    
    // Define the backward function in the module
    m.def("backward", &backward, "Flash Attention backward (CUDA)");
    
}
