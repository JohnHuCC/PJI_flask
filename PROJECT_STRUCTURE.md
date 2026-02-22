# Project Structure

## Core Runtime
- `app.py`: Flask + SocketIO main entry.
- `templates/`: HTML templates.
- `static/`: JS/CSS/assets.
- `model.py`, `db.py`: SQLAlchemy models and DB object.
- `docker-compose.yml`, `Dockerfile`: container runtime.

## Rule/Model Assets
- `Decision_rule/`: decision path JSON artifacts.
- `PJI_model/`, `Stacking_model/`, `Tree_Candidate/`: model and rule assets.

## Data
- `data/samples/Revision_PJI_test_2.csv`: sample CSV for local import/testing.
- `uploads/`: runtime upload output.
- `artifacts/`: runtime or generated artifacts (logs/reports/runtime temp files).

## Utility Scripts
- `scripts/generate_csv.py`: generate sample CSV.
- `scripts/pji_csv_to_db.py`: import revision CSV into MySQL.
- `scripts/pji_newdata_to_db.py`: legacy helper for message/new-data related table init.

## Archive
- `archive/`: old snapshots and legacy scripts kept for rollback/reference.
- `archive/app_old.py`: older app snapshot.
- `archive/app_recovery.py`: recovery version kept for reference.
- `archive/legacy/scripts/`: temporary or experimental scripts moved out of root.
- `archive/legacy/config/requirement.txt`: deprecated requirements file.

## Docs
- `docs/images/`: screenshots used by `README.md`.

## Common Commands
- Run app (docker): `docker compose up -d --build`
- Generate sample CSV: `python scripts/generate_csv.py`
- Import sample CSV: `python scripts/pji_csv_to_db.py`
