# 1. 基础系统
FROM python:3.8.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1.5 关键：由于 Debian 10 已退役，且国外官方档案馆极不稳定频繁报 503，
# 我们直接使用国内阿里云的【历史系统档案馆（debian-archive）】来实现极速、稳定下载
RUN echo "deb [trusted=yes] http://mirrors.aliyun.com/debian-archive/debian buster main" > /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.aliyun.com/debian-archive/debian-security buster/updates main" >> /etc/apt/sources.list && \
    echo "Acquire::Check-Valid-Until \"false\";" > /etc/apt/apt.conf.d/10no-check-valid-until

# 2. 安装机器人仿真和视频渲染依赖（去档案馆下载）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip

# 3. 安装显卡加速版 PyTorch 2.0.1
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 4. 安装 ACT 的运行依赖
RUN pip install dm_control==1.0.14 \
    mujoco==2.3.7 \
    h5py \
    einops \
    gym==0.21.0 \
    matplotlib \
    ipython \
    tensorboard

# 5. 将当前文件夹下的所有代码、模型权重拷贝进容器
COPY . /workspace

CMD ["/bin/bash"]
