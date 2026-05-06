FROM python:3.8-slim-bookworm

WORKDIR /app

COPY . /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    awscli \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]