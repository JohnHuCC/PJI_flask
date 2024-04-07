# 使用 Python 基礎映像檔
FROM python:3.9

# 設定工作目錄
WORKDIR /usr/src/app

# 安裝依賴
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製全部源代碼
COPY . .

# 設定環境變數
ENV FLASK_APP=app.py

# 啟動應用
CMD ["flask", "run", "--host=0.0.0.0"]
