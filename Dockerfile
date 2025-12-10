# OpenYOLO3D with CUDA 11.3 support, conda-based
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# ----------------------------------------------------------------------
# 1. System deps
# ----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    libblas-dev \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# 2. Install Miniconda
# ----------------------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Ensure we use bash for conda commands
SHELL ["bash", "-lc"]

WORKDIR /workspace/OpenYOLO3D

# ----------------------------------------------------------------------
# 3. Copy environment.yml and create conda env
#    (Python 3.10.9 as specified)
# ----------------------------------------------------------------------
COPY environment.yml ./environment.yml

RUN conda env create -f environment.yml && \
    conda clean -afy

# Set default env
ENV CONDA_DEFAULT_ENV=openyolo3d
ENV PATH=/opt/conda/envs/openyolo3d/bin:$PATH

# Optional: keep pip/setuptools reasonable for old CUDA stack
RUN python -m pip install --upgrade "pip<24.0" && \
    python -m pip install "setuptools<=65.6.3" "wheel"

# ----------------------------------------------------------------------
# 4. Copy the project AFTER env creation to leverage Docker cache
# ----------------------------------------------------------------------
COPY . .

# ----------------------------------------------------------------------
# 5. Follow README: install torch, scatter, detectron2, Mask3D deps
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/Mask3D

# PyTorch + scatter (inside conda env)
RUN pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# Detectron2 pinned to commit used by authors
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' \
    --no-deps

# ----------------------------------------------------------------------
# 6. Third-party for Mask3D: MinkowskiEngine, ScanNet, pointnet2
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/Mask3D/third_party

RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas=openblas

RUN git clone https://github.com/ScanNet/ScanNet.git && \
    cd ScanNet/Segmentator && \
    git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && \
    make

RUN cd pointnet2 && python setup.py install

# ----------------------------------------------------------------------
# 7. Remaining Python deps exactly as README
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/Mask3D

RUN pip install --no-cache-dir pytorch-lightning==1.7.2 && \
    pip install --no-cache-dir \
        black==21.4b2 \
        cloudpickle==3.0.0 \
        future \
        hydra-core==1.0.5 \
        "pycocotools>=2.0.2" \
        pydot \
        iopath==0.1.7 \
        loguru \
        albumentations && \
    pip install --no-cache-dir .

# ----------------------------------------------------------------------
# 8. YOLO-World, mmdet, mmyolo, mmcv, open3d, etc.
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/YOLO-World
RUN pip install --no-cache-dir -e .

WORKDIR /workspace/OpenYOLO3D
RUN pip install --no-cache-dir \
        mmyolo==0.6.0 \
        mmdet==3.0.0 \
        plyfile \
        openmim && \
    pip install --no-cache-dir \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 && \
    mim install mmcv==2.0.0 && \
    pip install --no-cache-dir \
        open3d \
        pillow==9.1.0 \
        pyviz3d \
        supervision==0.19.0 \
        shapely \
        transformers==4.30.0

# ----------------------------------------------------------------------
# 9. (Optional but nice): download class-agnostic masks & checkpoints
#     If you want them baked into the image.
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D
RUN chmod +x scripts/get_class_agn_masks.sh scripts/get_checkpoints.sh || true && \
    ./scripts/get_class_agn_masks.sh || echo "get_class_agn_masks.sh failed (check manually)" && \
    ./scripts/get_checkpoints.sh || echo "get_checkpoints.sh failed (check manually)"

# ----------------------------------------------------------------------
# 10. Runtime
# ----------------------------------------------------------------------
ENV PYTHONPATH="/workspace/OpenYOLO3D:${PYTHONPATH}"
WORKDIR /workspace/OpenYOLO3D

CMD ["bash"]
