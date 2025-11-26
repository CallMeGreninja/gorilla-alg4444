FROM nvcr.io/nvidia/pytorch:22.09-py3 

ENV PATH="/usr/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py utils.py dataset.py inference.py ./

ENTRYPOINT ["python", "inference.py"]