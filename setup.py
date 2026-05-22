from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eguzklean",
    version="0.1.0",
    description="Funciones auxiliares para EDA, visualización y preprocesamiento",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="eguzki-dm",
    author_email="eguzkia.lab@gmail.com",
    url="https://github.com/eguzki-dm/Eguzklean",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3",
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "seaborn>=0.11",
    ],
)
