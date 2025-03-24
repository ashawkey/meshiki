import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="meshiki",
    version="0.0.5",
    description="Unusual mesh processing tools",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashawkey/meshiki",
    author="kiui",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    ext_modules=[
        Pybind11Extension(
            name="_meshiki",
            sources=["src/bindings.cpp"], # just cpp files
            include_dirs=[os.path.join(_src_path, "include")],
            extra_compile_args=["-std=c++17", "-O3"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy", "pybind11", "trimesh", "kiui", "pymeshlab"],
)