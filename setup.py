from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, IS_WINDOWS, IS_MACOS

extra_compile_args = []
if IS_WINDOWS:
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/openmp')
elif IS_MACOS:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp=libomp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp')

setup(
    name='pycoriander',
    ext_modules=[
        CUDAExtension('pycoriander', [
            'src/Random/Random.cpp',
            'src/Random/Xorshift.cpp',
            'src/CudaHelpers.cpp',
            'src/MutualInformation.cpp',
            'src/MutualInformationCpu.cpp',
            'src/MutualInformationCuda.cpp',
        ], libraries=['nvrtc'], extra_compile_args=extra_compile_args)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
