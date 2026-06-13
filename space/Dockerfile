FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.huggingface \
    PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY affinity /app/affinity
COPY app.py /app/app.py

EXPOSE 7860
CMD ["python", "app.py"]
