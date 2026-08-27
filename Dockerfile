FROM python:3.9

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# 避免 pyeda 編譯時因為 qsort 指標型別警告被當成 error
ENV CFLAGS="-Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
EXPOSE 5000

CMD ["flask", "run"]
