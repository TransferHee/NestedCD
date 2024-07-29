from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="EarthMoverDistance",
    ext_modules=[CUDAExtension("EarthMoverDistance", ["emd.cpp", "emd_cuda.cu",]),],
    cmdclass={"build_ext": BuildExtension},
)

