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

## Main Pages

After login:

- `/` : Patient table
- `/personal_info?p_id=<ID>` : Personal info + model result
- `/model_diagnosis?p_id=<ID>` : Decision path page
- `/reactive_diagram?p_id=<ID>` : Reactive diagram page
- `/train_new_data` : New data table
- `/upload_new_data_csv` : Upload CSV/XLSX
- `/message_board` : Message board

---

## Page Walkthrough

### 1. Login Page (`/auth_login`)

- Entry point for the system.
- Use test account to sign in quickly.

![Login](docs/images/login.png)

### 2. Patient Table (`/`)

- Shows all patient records currently available in `revision_pji`.
- Supports search, paging, and direct navigation to diagnosis-related pages.
- Action buttons route to:
  - Personal info
  - Model diagnosis
  - Reactive diagram

![Patient Table](docs/images/index.png)

### 3. Personal Info (`/personal_info?p_id=<ID>`)

- Displays selected patient key fields and a model result summary (`Infected!` / `Aseptic!`).
- Used as the first-level view for one patient.

![Personal Info - Top](docs/images/basic_data_1.png)
![Personal Info - Feature Block](docs/images/basic_data_2.png)
![Personal Info - Full Fields](docs/images/basic_data_3.png)

### 4. Model Diagnosis (`/model_diagnosis?p_id=<ID>`)

- Visualizes decision paths used for model interpretation.
- Shows path graph and textual rule list.
- Slider controls how many paths/rules are displayed.

![To Diagnosis](docs/images/to_diagnosis.png)
![Diagnosis Graph](docs/images/diagnosis_1.png)
![Diagnosis Rules](docs/images/diagnosis_2.png)

### 5. Reactive Diagram (`/reactive_diagram?p_id=<ID>`)

- Interactive prediction view with feature-level values and model outputs.
- Displays base-model/meta-model flow and decision-related structures.
- Useful for scenario analysis and threshold checking.

![To Reactive Diagram](docs/images/to_interactive_prediction.png)
![Reactive Input](docs/images/interactive_pred_1.png)
![Reactive Progress](docs/images/interactive_pred_2.png)
![Reactive Graph](docs/images/interactive_pred_3.png)

### 6. New Data Flow

- `/upload_new_data_csv`: upload new CSV/XLSX data.
- `/train_new_data`: browse imported new data.
- `/new_data_buffer`: waiting pool for data transition/review.

This flow is used when adding fresh clinical records into the system pipeline.

### 7. Message Board (`/message_board`)

- Simple feedback/notes page.
- Saves message content into `message` table.

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
