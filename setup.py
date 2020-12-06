import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

scripts = [
    "bin/lpf",
]

setuptools.setup(
    name="lpf-druhe",
    version="0.0.1",
    author="David Ruhe",
    author_email="d.ruhe@uva.nl",
    description="LOFAR Pulse Finder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/transientskp/lpf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
