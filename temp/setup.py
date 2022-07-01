import setuptools


requires = [
    row.replace("\n", "").replace("\r\n", "")
    for row in open("requirements-dev.txt")
]





setuptools.setup(
    name="plastic-origins",
    version="0.0.7",
    author="Plastic Origins",
    author_email="chayma.mesbahi@neoxia.com",
    description="A package containing methods commonly used to make inferences",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Chayma-mesbahi/surfnet.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requires,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*model.xlsx", "../requirements-dev.txt", "../README.md"],
    },
)