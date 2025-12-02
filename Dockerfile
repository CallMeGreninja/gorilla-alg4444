FROM nvcr.io/nvidia/pytorch:22.09-py3

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopenslide0 \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py ./

COPY model_weights/best_model.pth /app/best_model.pth
RUN mkdir -p /opt/ml/model && cp /app/best_model.pth /opt/ml/model/

RUN mkdir -p /input /output /opt/algorithm && chmod 777 /input /output /app

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN chown -R algorithm:algorithm /app /opt/ml /opt/algorithm /input /output

USER algorithm

ENTRYPOINT ["python", "inference.py"]