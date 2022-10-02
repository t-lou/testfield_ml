FROM ubuntu:22.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install python3-pip clang-14 g++-12 wget unzip cmake build-essential -y && \
    apt-get clean && \
    useradd -m usr && echo "usr:usr" | chpasswd && \
    pip3 install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 cache purge && \
    cd /opt && \
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip -O output.zip && \
    unzip output.zip -d /opt && \
    cp -r /opt/libtorch/* /usr && \
    rm -rf *

USER usr