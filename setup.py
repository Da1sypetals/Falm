from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            },
            include_dirs=['include']  # Add this line if you have header files in the include directory
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
