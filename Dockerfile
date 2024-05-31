FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget && \
    mkdir -p /app/models && \
    wget -O /app/models/finetuned-deberta-v3-small-best.pth.tar https://github.com/hsushuai/detect-ai-generated-text/releases/download/models/finetuned-deberta-v3-small-best.pth.tar

EXPOSE 5000

CMD ["python", "deployment/app.py"]