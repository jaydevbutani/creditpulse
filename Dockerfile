# CreditPulse serving image: FastAPI scoring service (8000) + Streamlit dashboard (8501).
# Build after `make all` so artifacts/ and data/processed/ exist (or run `make all` inside the image).
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
COPY pyproject.toml Makefile ./
COPY configs configs
COPY src src
COPY app app
COPY scripts scripts
COPY tests tests
RUN pip install -e .
COPY data data
COPY artifacts artifacts
COPY reports reports
EXPOSE 8000 8501
HEALTHCHECK CMD curl -fs http://localhost:8000/health || exit 1
CMD ["sh", "-c", "uvicorn creditpulse.serve.api:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0"]
