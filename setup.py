import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SenseTheFlow-gpascualg", # Replace with your own username
    version="0.2.4.1",
    author="gpascualg",
    author_email="guillem.pascual@ub.edu",
    description="A tensorflow models library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpascualg/SenseTheFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
