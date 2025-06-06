# setup.py
from setuptools import setup, find_packages

setup(
    name="dl_comm",
    version="0.1.0",
    description="DL COMM: a lightweight benchmark for deep-learning communication patterns",
    author="Musa Cim",
    author_email="mcim@anl.gov",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "mpi4py>=3.0.0",
        "torch>=1.13.0",
        "hydra-core>=1.1.0",
        
    ],
    packages=find_packages(where="src"),    
    package_dir={"": "src"},                 
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "dl_comm = dl_comm.ml_comm:main"
        ]
    },
)
