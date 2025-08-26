from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob


root_path = os.path.dirname(__file__)
print(root_path)



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




def get_package_data():
    ext_src = []
    for ext in ["cpp", "cu", "h", "cuh"]:
        ext_src.extend(
            glob.glob(os.path.join("SAFFRON", "**", f"*.{ext}"), recursive=True)
        )
    return {"SFARRON": ["py.typed"] + [os.path.join("..", path) for path in ext_src]}



setup(
    name="saffron",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},

)
