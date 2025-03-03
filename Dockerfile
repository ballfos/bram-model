ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV TZ=Asia/Tokyo

RUN apt-get update \
&& apt-get install -y \
python3 \
python3-pip \
fonts-noto-cjk \
wget \
git \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN python3 -m pip install -r requirements.txt

CMD ["sleep", "infinity"]