FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[server]'
ENV TTS_MODEL=/models/en_US-lessac-medium.onnx
EXPOSE 8000
CMD ["uvicorn", "audio_book.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
