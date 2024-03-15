from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"
long_description = "All information are available here: https://github.com/StefanKapunac/ksrdp"

ext_modules = [
    Pybind11Extension(
        "ksrdp",
        sources=["src/vns.cpp", "src/greedy.cpp", "src/utils.cpp", "src/python_binding.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        # extra_compile_args=['-std=c++17', '-O3'],
        # include_dirs=["src/"],
    ),
]

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)