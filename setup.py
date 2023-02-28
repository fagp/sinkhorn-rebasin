import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rebasin",
    version="0.0.1a",
    author="Fidel Guerrero Pena",
    author_email="fidel-alejandro.guerrero-pena@etsmtl.ca",
    description="Python package for differentiable re-basin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "matplotlib",
        "tqdm",
        "torchviz",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
)
