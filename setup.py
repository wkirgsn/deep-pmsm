import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-pmsm",
    version="0.0.1",
    author="Wilhelm Kirchgässner",
    author_email="kirchgaessner@lea.upb.de",
    description="Estimate PMSM temperatures with RNNs and CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wkirgsn/deep-pmsm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
