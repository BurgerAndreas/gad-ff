from setuptools import find_packages, setup


setup(
    name="gadff",
    version="1.0.0",
    # packages=find_packages(),
    packages=['gadff'],
    install_requires=['h5py', 'progressbar'],
    extras_require={
        "example": ["ase","tqdm"],
    },
)
