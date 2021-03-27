from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='QGTC',
    ext_modules=[
        CUDAExtension(name='QGTC', 
            sources=[
            'QGTC_host.cpp',
            'QGTC_device.cu'
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-arch=sm_86']}
         ) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    })