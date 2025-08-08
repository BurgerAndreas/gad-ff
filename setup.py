import setuptools


setuptools.setup(
    name="gadff",
    version="1.0.0",
    # packages=setuptools.find_packages(),
    packages=[
        "gadff",
        "nets",
        "ocpmodels",
        "horm_alphanet",
        "horm_leftnet",
        "recipes",
        "scripts",
    ],
)
