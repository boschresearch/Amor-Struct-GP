from setuptools import find_packages, setup

from amorstructgp import __version__

name = "amorstructgp"
version = __version__
description = "Amortized Inference for Gaussian Process Hyperparameters of Structured Kernels"

setup(name=name, version=version, packages=find_packages(exclude=["tests"]), description=description)
