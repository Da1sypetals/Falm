#ifndef KERNEL_H
#define KERNEL_H

#include <torch/extension.h>

void lanuch_forward_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor l, torch::Tensor m, torch::Tensor O);

#endif 
