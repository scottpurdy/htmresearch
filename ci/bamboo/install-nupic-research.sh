#!/bin/bash
set -o errexit
set -o xtrace

# Setup compiler
export CC="clang"
export CXX="clang++"

# Install OS dependencies, assuming stock ubuntu:latest
apt-get update
apt-get install -y \
    wget \
    git \
    build-essential \
    clang \
    clang++ \
    python2.7-dev \
    python-pip
pip install --upgrade setuptools
pip install --upgrade pip
pip install wheel

# Install nupic wheel and dependencies, including nupic.bindings artifact in
# wheelwhouse/
pip install -f wheelhouse/ nupic.bindings nupic

python setup.py install
