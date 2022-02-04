from setuptools import setup, Extension, find_packages
import numpy
import platform
import struct

import contextlib
import os

import sys

sys.command("$(brew --prefix openblas)")

# Override sdist to always produce .zip archive
from distutils.command.sdist import sdist as _sdist

class sdistzip(_sdist):
    def initialize_options(self):
        _sdist.initialize_options(self)
        self.formats = ['zip', 'gztar']
        
print(numpy.show_config())

if platform.system() == "Darwin":
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
    os.environ["LDFLAGS"] = "-L/usr/local/opt/openblas/lib"
    os.environ["CPPFLAGS"] = "-I/usr/local/opt/openblas/include"

def getBlas():
    file_ = open("npConfg_file.txt", "w")
    with contextlib.redirect_stdout(file_):
        numpy.show_config()
    file_.close()
    np_confg = open('npConfg_file.txt', 'r')
    lib = ""
    for line in np_confg:
        if 'libraries' in line:
            lib = line
            break
    np_confg.close()
    os.remove("npConfg_file.txt")
    if lib != "":
        blas = lib.split('[')[1].split(',')[0]
        return blas[1:len(blas) - 1]
    else:
        return lib


np_blas = getBlas()

LIBS = []
INCLUDE_DIRS = []
EXTRA_COMPILE_ARGS = []
LIBRARY_DIRS = []
RUNTIME_LIRABRY_DIRS = []

if platform.system() == "Windows":
    if 'mkl' in np_blas:
        libs_mkl_windows = ['mkl_rt', 'iomp5']
        include_dirs_mkl_windows = [numpy.get_include()]
        extra_compile_args_mkl_windows = [
            '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '/permissive-', '/W1']
        LIBS = libs_mkl_windows
        INCLUDE_DIRS = include_dirs_mkl_windows
        EXTRA_COMPILE_ARGS = extra_compile_args_mkl_windows

    if np_blas == "" or "openblas" in np_blas:
        extra_compile_args_open_blas = [
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '/PIC',
            '/permissive-', '/W1', '/openmp']
        libs_open_blas = ["libopenblas"]
        include_dirs_open_blas = [numpy.get_include()]

        LIBS = libs_open_blas
        INCLUDE_DIRS = include_dirs_open_blas
        EXTRA_COMPILE_ARGS = extra_compile_args_open_blas

    elif 'blas' in np_blas:
        extra_compile_args_open_blas = [
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '/PIC',
            '/permissive-', '/W1', '/openmp']
        libs_open_blas = np_blas
        include_dirs_open_blas = [numpy.get_include()]
        LIBS = libs_open_blas
        INCLUDE_DIRS = include_dirs_open_blas
        EXTRA_COMPILE_ARGS = extra_compile_args_open_blas

    if struct.calcsize("P") * 8 == 32:
        INCLUDE_DIRS = ['D:/a/cyanure/cyanure/openblas_86/include'] + INCLUDE_DIRS
        LIBRARY_DIRS = ['D:/a/cyanure/cyanure/openblas_86/lib']
        EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS
    else:
        INCLUDE_DIRS = ['D:/a/cyanure/cyanure/openblas_64/include'] + INCLUDE_DIRS
        LIBRARY_DIRS = ['D:/a/cyanure/cyanure/openblas_64/lib']
        EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS

else:
    ##### setup mkl_rt
    if 'mkl' in np_blas:
        extra_compile_args_mkl = [
            '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '-fPIC',
            '-fopenmp', '-std=c++11']
        libs_mkl = ['mkl_rt', 'iomp5']
        include_dirs_mkl = [numpy.get_include()]

        LIBS = libs_mkl
        INCLUDE_DIRS = include_dirs_mkl
        EXTRA_COMPILE_ARGS = extra_compile_args_mkl

    ##### setup openblas
    if 'blas' in np_blas:
        
        if 'openblas' in np_blas:
            libs_open_blas = ['openblas']
        else:
            libs_open_blas = [np_blas]

        extra_compile_args_open_blas = [
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC',
            '-std=c++11', '-fopenmp']      

        include_dirs_open_blas = [numpy.get_include()]

        LIBS = libs_open_blas
        INCLUDE_DIRS = include_dirs_open_blas
        EXTRA_COMPILE_ARGS = extra_compile_args_open_blas

        INCLUDE_DIRS = ['/usr/local/opt/openblas/include'] + INCLUDE_DIRS
        LIBRARY_DIRS = ['/usr/local/opt/openblas/lib']
        LIBS = LIBS
        RUNTIME_LIRABRY_DIRS = LIBRARY_DIRS
        EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS

        if platform.system() == "Darwin":
            INCLUDE_DIRS = ["/usr/local/include", "/usr/local/opt/llvm/include"] + INCLUDE_DIRS
            LIBRARY_DIRS = ["/usr/local/lib", "/usr/local/opt/llvm/lib"] + LIBRARY_DIRS

print("DEBUG INSTALL: " + np_blas)
"""
## setup openblass no openmp

libs_open_blass_no_openmp = ['openblas']
include_dirs_open_blass_no_openmp = [numpy.get_include()]
extra_compile_args_open_blass_no_openmp =[
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC', '-std=c++11']

#### setup mkl no openmp
libs_mkl_no_openmp = ['mkl_rt']
include_dirs_mkl_no_openmp = [numpy.get_include()]
extra_compile_args_mkl_no_openmp =[
                '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '-fPIC',
                '-std=c++11']
n argumentss

"""

if platform.system() != "Windows":
    EXTRA_LINK_ARGS = ['-fopenmp']
else:
    EXTRA_LINK_ARGS = []
    

cyanure_wrap = Extension(
    'cyanure_lib.cyanure_wrap',
    libraries=LIBS,
    include_dirs=INCLUDE_DIRS,
    language='c++',
    library_dirs=LIBRARY_DIRS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    runtime_library_dirs=RUNTIME_LIRABRY_DIRS,
    extra_link_args=EXTRA_LINK_ARGS,
    sources=['cyanure_lib/cyanure_wrap_module.cpp'])

setup(name='cyanure',
      version='0.22.4',
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="http://julien.mairal.org/cyanure/",
      description='optimization toolbox for machine learning',
      install_requires=['scipy', 'numpy>=1.18.0', 'scikit-learn'],
      ext_modules=[cyanure_wrap],
      packages=find_packages(),
      cmdclass={'sdist': sdistzip},
      py_modules=['cyanure'])
