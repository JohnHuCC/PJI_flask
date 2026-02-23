# PJI Flask System

PJI Flask is a web system for prosthetic joint infection (PJI) pre-diagnosis.
It provides:

- Patient list and personal information page
- Model diagnosis view (decision paths)
- Reactive diagram view
- New-data upload and training-data pool pages

## Quick Start (Recommended: Docker)

### 1. Clone repository

```bash
git clone https://github.com/JohnHuCC/PJI_flask.git
cd PJI_flask
```

### 2. Start services

```bash
docker compose up -d --build
```

This starts:

- `web` on `http://127.0.0.1:5001`
- `mysql` on `127.0.0.1:3306`

### 3. Login

Open:

- `http://127.0.0.1:5001/auth_login`

Default test account:

- Username: `test`
- Password: `1234`

---

## Page Function Map (Chinese + English)

This section describes **every active page route** in the current system.

### Authentication / 驗證

1. `/auth_login`
- 中文: 系統登入入口頁，輸入帳號密碼後進入主系統。
- English: System login entry. Submit username/password to access the app.
- Notes: Most routes require login; unauthenticated requests are redirected here.

2. `/logout`
- 中文: 清除目前 session，回到登入頁。
- English: Clear current session and return to login page.

![Login](docs/images/login.png)

### Core Clinical Pages / 核心臨床頁面

1. `/`
- 中文: 病患總表（資料來源 `revision_pji`），可搜尋、分頁，並導向個案頁。
- English: Patient master table (from `revision_pji`) with search/pagination and links to per-patient pages.
- Actions: 個人資料 / 模型診斷 / 互動圖表。

2. `/personal_info?p_id=<ID>`
- 中文: 顯示單一病患重點欄位與模型判斷摘要，作為個案主檢視頁。
- English: Patient detail page with key features and model summary result.

3. `/model_diagnosis?p_id=<ID>`
- 中文: 顯示 decision path 圖與規則文字；可用滑桿控制顯示路徑數量。
- English: Decision-path interpretation page with graph and textual rules; slider controls number of displayed paths.

4. `/reactive_diagram?p_id=<ID>`
- 中文: 互動式特徵調整與模型結果觀察頁，包含流程圖與簡化規則。
- English: Interactive scenario-analysis page for feature tuning, prediction flow, and simplified rules.

![Patient Table](docs/images/index.png)
![Personal Info - Top](docs/images/basic_data_1.png)
![Diagnosis Graph](docs/images/diagnosis_1.png)
![Reactive Graph](docs/images/interactive_pred_3.png)

### New Data Pipeline / 新資料流程

1. `/upload_new_data_csv`
- 中文: 上傳 CSV/XLSX 新資料，前端先做欄位驗證，再送入新資料池。
- English: Upload CSV/XLSX and validate required columns before importing into new-data pool.

2. `/train_new_data`
- 中文: 新資料清單頁（`pji_new_data`），可對單筆資料進行模型預覽流程。
- English: New-data list (`pji_new_data`) for reviewing recently imported records.

3. `/pick_new_data?p_id=<ID>`
- 中文: 新資料「單筆預覽頁」；用途是確認該病患資料與規則，之後可移至 waiting pool。
- English: Single-record preview for new data; inspect model/rules before moving to waiting pool.

4. `/upload_new_data?p_id=<ID>`
- 中文: 動作路由，將資料由 `pji_new_data` 移到 `pji_new_data_buffer`。
- English: Action route to move one record from `pji_new_data` to `pji_new_data_buffer`.
- Notes: 建議由按鈕觸發，不是給一般使用者當內容頁直接瀏覽。

5. `/new_data_buffer`
- 中文: 等待池清單頁（`pji_new_data_buffer`），可查閱、退回或合併。
- English: Waiting-pool list (`pji_new_data_buffer`) for review, return, or merge operations.

6. `/pick_new_data_view?p_id=<ID>`
- 中文: waiting pool 的單筆預覽頁，確認後可退回新資料池或執行合併。
- English: Single-record preview in waiting pool before return/merge decision.

7. `/back_new_data?p_id=<ID>`
- 中文: 動作路由，將資料由 `pji_new_data_buffer` 退回 `pji_new_data`。
- English: Action route to move one record back from waiting pool to new-data pool.

8. `/merge_new_data`
- 中文: 動作路由，將 waiting pool 全部資料合併進 `pji_new_data`，並清空 waiting pool。
- English: Action route to merge all waiting-pool records into `pji_new_data` and clear buffer.

### Misc / 其他功能

1. `/message_board`
- 中文: 留言頁，提交訊息到 `message` 資料表。
- English: Message page that stores feedback/notes in `message` table.

2. `/adjust_rate_limiting`
- 中文: 調整 rate limit 參數的管理路由（非一般使用者常用頁面）。
- English: Admin/maintenance route for rate-limiting adjustment.

### Route Usage Note / 路由使用說明

- 中文: `upload_new_data`、`back_new_data`、`merge_new_data` 屬於「動作型路由」，主要由按鈕或程式流程觸發。
- English: `upload_new_data`, `back_new_data`, and `merge_new_data` are action routes, intended to be triggered by UI actions/workflow, not as standalone content pages.

---

## Import Sample Data

Generate sample CSV:

```bash
python scripts/generate_sample_revision_csv.py
```

Import sample CSV into database:

```bash
python scripts/import_revision_csv_to_db.py
```

Default sample path:

- `data/samples/Revision_PJI_test_2.csv`

---

## Local Run (without Docker)

Use this only if you already have MySQL and Python environment ready.

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Example:

```bash
export DATABASE_URL='mysql+pymysql://root:YOUR_PASSWORD@127.0.0.1:3306/PJI'
export MYSQL_HOST='127.0.0.1'
export MYSQL_PORT='3306'
export MYSQL_USER='root'
export MYSQL_PASSWORD='YOUR_PASSWORD'
export MYSQL_DB='PJI'
export APP_HOST='127.0.0.1'
export APP_PORT='5001'
export APP_DEBUG='false'
```

### 4. Run

```bash
python app.py
```

---

## Project Layout

- `app.py` : Flask main app
- `templates/` : HTML templates
- `static/` : JS/CSS/assets
- `Decision_rule/` : decision rule JSON artifacts
- `scripts/` : utility scripts
- `data/samples/` : sample CSV data
- `archive/` : legacy and reference files
- `docs/images/` : screenshots used in docs

---

## Common Commands

Rebuild web service:

```bash
docker compose up -d --build web
```

Check service status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f web
```

Stop services:

```bash
docker compose down
```

---

## Troubleshooting

### Login page keeps loading

- Make sure containers are up: `docker compose ps`
- Check web logs: `docker compose logs -f web`

### Internal Server Error after login

- Usually caused by missing DB table or bad DB connection.
- Confirm MySQL env in `docker-compose.yml` is correct.

### Page has no data

- Import sample data:
  - `python scripts/generate_sample_revision_csv.py`
  - `python scripts/import_revision_csv_to_db.py`

### Port conflict

If `5001` is occupied, update `docker-compose.yml` mapping and `APP_PORT` accordingly.

---

## Notes

- This project includes archived historical scripts in `archive/` for traceability.
- Core runtime is centered on `app.py` and routes under current templates/static assets.
