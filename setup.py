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
                'cxx': ['-O1'],  # Any additional C++ compiler flags
                'nvcc': ['-O1', '--threads=8'] 
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(parallel=True)
    }
)
