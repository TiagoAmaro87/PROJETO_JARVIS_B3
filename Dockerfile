FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: dry-run mode
ENV JARVIS_MODE=dry_run
CMD ["python", "BRAIN (DL)/brain_engine_v2.py"]
