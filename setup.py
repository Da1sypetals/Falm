from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attn', # Package name
    ext_modules=[
        CUDAExtension(
            name='flash_attn', # Module extension name
            sources=[
                'src/flash_attn.cpp',
                # 'kernels/forward_kernel.cu',
                # 'kernels/backward_kernel.cu'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
