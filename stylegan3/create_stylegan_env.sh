#!/bin/bash

echo "Creating environment..."
conda create -n stylegan3 python=3.9 -y

echo "Installing PyTorch and CUDA Toolkit..."
conda install -n stylegan3 -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing Compatible GCC/G++ Compilers..."
conda install -n stylegan3 -y -c conda-forge "gxx_linux-64=11" "gcc_linux-64=11" "sysroot_linux-64=2.17"

echo "Installing CUDA Toolkit with NVCC Compiler..."
conda install -n stylegan3 -y -c nvidia cuda-toolkit=11.8

echo "Installing CUDA-CCCL..."
conda install -n stylegan3 -y -c nvidia cuda-cccl=11.8

echo "Installing CUDA dev libraries..."
conda install -n stylegan3 -y -c nvidia cuda-libraries-dev=11.8

echo "Installing Ninja..."
conda install -n stylegan3 -y -c conda-forge ninja

echo "Installing other dependencies..."
conda install -n stylegan3 -y psutil click scipy 

echo "Configuring persistent environment variables..."
conda env config vars set -n stylegan3 CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
conda env config vars set -n stylegan3 CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
conda env config vars set -n stylegan3 CPATH=$CONDA_PREFIX/include:$CPATH
conda env config vars set -n stylegan3 CUDA_HOME=$CONDA_PREFIX/envs/stylegan3
conda env config vars set -n stylegan3 LD_LIBRARY_PATH=$CONDA_PREFIX/envs/stylegan3/lib:$LD_LIBRARY_PATH

echo "Cleaning torch_extensions cache directory..."
rm -rf ~/.cache/torch_extensions/*

echo "Done! You can now run: conda activate stylegan3"
