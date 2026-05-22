from setuptools import setup, find_packages

setup(
    name="eguzklean",
    version="0.1.0",
    description="Funciones auxiliares para EDA, visualización y preprocesamiento",
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
