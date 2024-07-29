from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mchamfer_3D3',
    ext_modules=[
        CUDAExtension('mchamfer_3D3', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamferM3D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })