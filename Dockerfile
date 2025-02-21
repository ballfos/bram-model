FROM python:3.12

ENV TZ=Asia/Tokyo

WORKDIR /workspace

COPY . .

RUN apt-get update \
 && apt-get install -y \
    fonts-noto-cjk \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -r requirements.txt

CMD ["sleep", "infinity"]