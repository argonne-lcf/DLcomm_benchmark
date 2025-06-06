# setup.py
import pathlib
from setuptools import setup, find_packages

 
long_description = (pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="dl_comm",
    version="0.1.0",
    description="DL COMM: collective communication benchmark for dl workloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Musa Cim",
    author_email="mcim@anl.gov",
    license="Apache-2.0",
    python_requires=">=3.8",

 
    packages=find_packages(where="."),      
    package_dir={"": "."},                   

 
    include_package_data=True,

    install_requires=[
        "mpi4py>=3.0.0",
        "torch>=1.13.0",
        "hydra-core>=1.1.0",
    ],

 
    entry_points={
        "console_scripts": [
            "dl_comm = dl_comm.ml_comm:main",
        ],
    },

 
 
)
