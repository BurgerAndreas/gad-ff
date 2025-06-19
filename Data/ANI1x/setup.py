from setuptools import find_packages, setup


setup(
    name="ani1x",
    version="1.0.0",
    # packages=find_packages(),
    packages=["ani1x"],
    install_requires=["h5py", "progressbar"],
)
