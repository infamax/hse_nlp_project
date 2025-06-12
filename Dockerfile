FROM ubuntu:22.04

# install python3, git and other developer tools 
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        libglib2.0-0 \
        libnss3 \
        libgconf-2-4 \
        libfontconfig1 \
        git \
        python3 \
        python3-pip \
    && rm -rf var/lib/apt/lists/*

# install selenium chrome for parsing
RUN apt-get update \
    && apt-get install -y \
        chromium-browser \
        chromium-chromedriver \
    && rm -rf var/lib/apt/lists/*


WORKDIR /workspace

COPY . /workspace/



