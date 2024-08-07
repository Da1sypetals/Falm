import os
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

include_dir = os.path.abspath('include')

setup(
    name='flash_attn', # Package name
    ext_modules=[
        CUDAExtension(
            name='flash_attn', # Module extension name
            sources=[
                'src/flash_attn.cpp',
                'kernels/forward_kernel.cu',
                'kernels/backward_kernel.cu'
            ],
            include_dirs=[include_dir],
            extra_compile_args={
                'cxx': [],  # Any additional C++ compiler flags
                'nvcc': ['-DTORCH_USE_CUDA_DSA']  # Enable device-side assertions
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
