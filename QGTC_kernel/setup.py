from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GAcc',
    ext_modules=[
        CUDAExtension('GAcc', [
            'QGTC_host.cpp',
            'QGTC_device.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })