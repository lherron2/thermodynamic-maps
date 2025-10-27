from setuptools import setup, find_packages
import os

setup(
    name="thermodynamic-maps",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    author="Lukas Herron",
    author_email="lherron@umd.edu",
    python_requires='>=3.10',
)

