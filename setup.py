from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob


root_path = os.path.dirname(__file__)
print(root_path)


# version: dict = dict()
# with open("./nesvor/version.py") as fp:
#     exec(fp.read(), version)


# def get_long_description():
#     with open("README.md", "r") as fh:
#         long_description = fh.read()
#     return long_description


def get_extensions():
    extensions = [
        CUDAExtension(
            name="saffron.slice_acq_cuda",
            sources=[
                os.path.join(
                    root_path, "saffron", "slice_acquisition", "slice_acq_cuda.cpp"
                ),
                os.path.join(
                    root_path, "saffron", "slice_acquisition", "slice_acq_cuda_kernel.cu"
                ),
            ],
        ),
        CUDAExtension(
            name="saffron.transform_convert_cuda",
            sources=[
                os.path.join(
                    root_path, "saffron", "transform", "transform_convert_cuda.cpp"
                ),
                os.path.join(
                    root_path, "saffron", "transform", "transform_convert_cuda_kernel.cu"
                ),
            ],
        ),
    ]
    return extensions


# def get_extensions():
#     extensions = [
#         CUDAExtension(
#             name="dnesvor.slice_acq_cuda",
#             sources=[
#                 os.path.join(
#                     root_path, "dnesvor", "slice_acquisition", "slice_acq_cuda.cpp"
#                 ),
#                 os.path.join(
#                     root_path, "dnesvor", "slice_acquisition", "slice_acq_cuda_kernel.cu"
#                 ),
#             ],
#         ),
#         CUDAExtension(
#             name="dnesvor.transform_convert_cuda",
#             sources=[
#                 os.path.join(
#                     root_path, "dnesvor", "transform", "transform_convert_cuda.cpp"
#                 ),
#                 os.path.join(
#                     root_path, "dnesvor", "transform", "transform_convert_cuda_kernel.cu"
#                 ),
#             ],
#         ),
#     ]
#     return extensions

def get_package_data():
    ext_src = []
    for ext in ["cpp", "cu", "h", "cuh"]:
        ext_src.extend(
            glob.glob(os.path.join("SAFFRON", "**", f"*.{ext}"), recursive=True)
        )
    return {"SFARRON": ["py.typed"] + [os.path.join("..", path) for path in ext_src]}


# def get_entry_points():
#     entry_points = {
#         "console_scripts": ["nesvor=nesvor.cli.main:main"],
#     }
#     return entry_points


setup(
    name="saffron",
    # packages=find_packages(exclude=("tests",)),
    # version=version["__version__"],
    # description="SAFFRON: toolkit for Gaussian slice-to-volume reconstruction",
    # long_description=get_long_description(),
    # long_description_content_type="text/markdown",
    # url=version["__url__"],
    # author=version["__author__"],
    # author_email=version["__email__"],
    # license="MIT",
    # zip_safe=False,
    # entry_points=get_entry_points(),
    ext_modules=get_extensions(),
    # package_data=get_package_data(),
    cmdclass={"build_ext": BuildExtension},
    # classifiers=[
    #     # "Intended Audience :: Healthcare Industry",
    #     # "Intended Audience :: Science/Research",
    #     # "License :: OSI Approved :: MIT License",
    #     # "Topic :: Scientific/Engineering :: Medical Science Apps.",
    #     # "Topic :: Scientific/Engineering :: Artificial Intelligence",
    #     # "Topic :: Scientific/Engineering :: Image Processing",
    #     "Environment :: GPU :: NVIDIA CUDA",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: C++",
    # ],
)
