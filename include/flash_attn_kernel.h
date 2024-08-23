#ifndef KERNEL_H
#define KERNEL_H

#include <torch/extension.h>

void launch_forward_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor l, torch::Tensor m,
                           torch::Tensor O);
void launch_backward_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor dQ,
                            torch::Tensor dK, torch::Tensor dV, torch::Tensor dO, torch::Tensor l, torch::Tensor m);

#endif
