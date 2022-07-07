FROM gcr.io/deeplearning-platform-release/sklearn-cpu.0-23
# FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest
# FROM python:3.10-slim
WORKDIR /app
COPY . .

# RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3","model_training.py"]
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app