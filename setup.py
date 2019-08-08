import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "tfExperiment",
    version = "1.0.1",
    author = "Pedro A. Favuzzi",
    author_email = "pa.favuzzi@gmail.com",
    description = "A simple library to manage Tensorflow experiments though git and reduce boilerplate. Compatible with tf 1.x",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Pensarfeo/tfExperiment",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [            # I get to this in a second
        "dotmap",
        "dataSaver",
        "stepTimer",
        "pygit2",
    ],
)