# Container for hosting the REST API (Render, or any Docker host).
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small (the CUDA build is several GB).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MINIGPT_CKPT=checkpoints/mini_gpt_best.pt
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
