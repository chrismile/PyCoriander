import sys
from setuptools import setup
from setuptools.command.egg_info import egg_info
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

class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)

setup(
    name='pycoriander',
    author='Christoph Neuhauser',
    ext_modules=[
        CUDAExtension('pycoriander', [
            'src/Random/Random.cpp',
            'src/Random/Xorshift.cpp',
            'src/CudaHelpers.cpp',
            'src/Correlation.cpp',
            'src/MutualInformation.cpp',
            'src/PyCorianderCpu.cpp',
            'src/PyCorianderCuda.cpp',
        ], libraries=['nvrtc'], extra_compile_args=extra_compile_args)
    ],
    data_files=[
        ( '.', ['src/pycoriander.pyi'] )
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'egg_info': EggInfoInstallLicense
    },
    license_files = ('LICENSE',)
)
