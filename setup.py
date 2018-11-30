# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from os.path import expanduser, dirname, join, abspath
from glob import glob
from itertools import chain
from subprocess import check_output, CalledProcessError
import sys

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

sources = list(chain(
    glob('*.cpp'),
    glob('featurizers/*.cpp'),
    glob('TorchCraft/client/*.cpp'),
    glob('TorchCraft/replayer/*.cpp'),
))
print(sources)

ext_modules = [
    Extension(
        '_ext',
        sources,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            "TorchCraft/include",
            "TorchCraft/include/torchcraft",
            "TorchCraft/replayer",
            "TorchCraft/BWEnv/fbs",
            "TorchCraft",
            ".",
        ],
        # TODO Dynamically search for this somehow???
        define_macros=[('WITH_ZSTD', None)],
        libraries=['zstd', 'zmq'],
        language='c++'
    ),
]

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for arg in sys.argv[1:]:  # additional args for compilation? (e.g. -I)
            opts.append(arg)
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++11")
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='_ext',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.1'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    script_name='setup.py',
    script_args=['build_ext', '--inplace']
)
