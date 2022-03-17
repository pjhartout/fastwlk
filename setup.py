#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Philip Hartout",
    author_email="philip.hartout@protonmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    description="fastwlk is a Python package that implements a fast version of the Weisfeiler-Lehman kernel.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords="fastwlk",
    name="fastwlk",
    packages=find_packages(include=["fastwlk", "fastwlk.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/pjhartout/fastwlk",
    version="0.1.0",
    zip_safe=False,
)
