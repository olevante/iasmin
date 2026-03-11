FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (for playwright + build)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        libglib2.0-0 \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpangocairo-1.0-0 \
        libpango-1.0-0 \
        libgtk-3-0 \
        libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m playwright install chromium

COPY . ./

EXPOSE 8501

CMD ["sh", "-c", "python -m streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
