# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, , 2018-11-12 20:14:49
# ***
# ************************************************************************************/
#

import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "imagecluster")
    sources = glob.glob(os.path.join(extensions_dir, "*.cpp"))

    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "imagecluster.cluster_cpp",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="imagecluster",
    version="1.0.0",
    description="Image Cluster Module",
    url="https://github.com/delldu/ImageCluster",
    author="Dell Du",
    packages=['imagecluster'],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
