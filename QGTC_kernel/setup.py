from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='QGTC',
    ext_modules=[
        CUDAExtension('QGTC', [
            'QGTC_host.cpp',
            'QGTC_device.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })