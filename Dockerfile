# This image is based on tensorflow/tensorflow-latest-gpy-py3,
# and fixes the CUDA and CDNN version
ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

LABEL maintainer="Kristian,Tobias"


# ARCH and CUDA are specified again because the FROM directive resets ARGs
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.4.38-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cuda-command-line-tools-${CUDA/./-} \
    # There appears to be a regression in libcublas10=10.2.2.89-1 which
    # prevents cublas from initializing in TF. See
    # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
    libcublas10=10.2.1.243-1 \ 
    cuda-nvrtc-${CUDA/./-} \
    cuda-cufft-${CUDA/./-} \
    cuda-curand-${CUDA/./-} \
    cuda-cusolver-${CUDA/./-} \
    cuda-cusparse-${CUDA/./-} \
    curl \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip \
    # following libraries are needed to run cv2
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx


# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
    apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
    libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# install python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# upgrade pip
RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# set working directory
WORKDIR  /srv/idp-radio-1

# add workdir to pythonpath
ENV PYTHONPATH "${PYTHONPATH}:/srv/idp-radio-1/src"

# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# install requirements for access jupyter lab and notebook remotely
RUN apt-get -y install git wget unzip ipython
RUN pip install ipython[notebook] jupyterlab

ENTRYPOINT ["./remote_access/open_remoteaccess.sh"]
CMD ["/bin/bash"]
