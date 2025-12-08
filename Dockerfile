# OpenYOLO3D with CUDA 11.3 support
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    ca-certificates \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libopenblas-dev \
    libblas-dev \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 3. DOWNGRADE PIP
RUN python -m pip install --upgrade "pip<24.0"

# 4. FIX DEPENDENCIES
RUN python -m pip install "setuptools==59.5.0" "wheel" "typing-extensions==4.9.0"

WORKDIR /workspace/OpenYOLO3D

# 5. Install PyTorch
RUN python -m pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
 && python -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# 6. Install MMLab ecosystem DIRECTLY
RUN python -m pip install --no-cache-dir "mmcv==2.0.0" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html && \
    python -m pip install --no-cache-dir \
      mmdet==3.0.0 \
      mmyolo==0.6.0

# 7. Install Python dependencies
RUN python -m pip install --no-cache-dir \
      pytorch-lightning==1.7.2 \
      hydra-core==1.0.5 \
      pycocotools>=2.0.2 \
      pydot \
      iopath==0.1.7 \
      loguru \
      albumentations \
      open3d \
      pillow==9.1.0 \
      plyfile \
      black==21.4b2 \
      cloudpickle==3.0.0 \
      future \
      pyviz3d \
      transformers==4.30.0 \
      shapely

# 8. Install Detectron2
RUN python -m pip install --no-cache-dir \
      'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' \
      --no-deps

# COPY project files
COPY . .

# 9. MANUALLY CLONE SUBMODULES
RUN rm -rf models/Mask3D/third_party/MinkowskiEngine && \
    git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" models/Mask3D/third_party/MinkowskiEngine && \
    cd models/Mask3D/third_party/MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas=openblas

RUN rm -rf models/Mask3D/third_party/ScanNet && \
    git clone https://github.com/ScanNet/ScanNet.git models/Mask3D/third_party/ScanNet && \
    cd models/Mask3D/third_party/ScanNet/Segmentator && \
    git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && \
    make

RUN cd models/Mask3D/third_party/pointnet2 && python setup.py install

# 10. Install the sub-projects
RUN python -m pip install --no-deps -e models/Mask3D
RUN python -m pip install --no-deps -e models/YOLO-World
# REMOVED: pip install . (No setup.py exists in root, handled by PYTHONPATH below)

ENV PYTHONPATH="/workspace/OpenYOLO3D:${PYTHONPATH}"

CMD ["bash"]