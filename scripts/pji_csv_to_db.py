#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from pathlib import Path

from sqlalchemy import create_engine, MetaData, Table, Column, String, text
from sqlalchemy.exc import IntegrityError


# ====== 你可以改這裡 ======
DEFAULT_DB_URL = "mysql+pymysql://root@127.0.0.1:3306/PJI"
DEFAULT_CSV_PATH = "data/samples/Revision_PJI_test_2.csv"
TABLE_NAME = "revision_pji"
TRUNCATE_BEFORE_IMPORT = False  # True = 每次匯入前先清空表（小心！）
ECHO_SQL = True
# ========================


# 欄位順序（你原本的 title）
TITLE = [
    "no_group", "no", "group", "name", "ctno", "csn", "PJI_revision_date",
    "Date_of_first_surgery", "Date_of_last_surgery", "primary_revision_native_hip",
    "acute_chronic", "asa", "laterality", "joint", "gender", "age", "height",
    "weight", "bmi", "serum_ESR", "serum_WBC", "segment", "hgb", "platelet",
    "p_t", "aptt", "fibrinogen", "d_dimer", "Serum_CRP", "cr", "ast", "alt",
    "positive_histology", "synovial_WBC", "synovial_Neutrophil", "turbidity",
    "color", "lymphocyte", "monocyte", "eosinophil", "synovial_leukocyte_esterase",
    "total_core", "icm_2", "minor_icm_criteria_total", "icm_1",
    "minor_msis_criteria_total", "msis_final_classification", "positive_culture",
    "sinus_tract", "pulurence", "single_positive_culture", "total_cci",
    "congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease",
    "pulmonary_circulation_disorders", "peripheral_vascular_disorders",
    "hypertension_uncomplicated", "hypertension_complicated", "paralysis",
    "other_neurological_disorders", "chronic_pulmonary_disease",
    "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism",
    "renal_failure", "liver_disease", "peptic_ulcer_disease_excluding_bleeding",
    "aids_hiv", "lymphoma", "metastatic_cancer", "solid_tumor_without_metastasis",
    "rheumatoid_arthritis_collagen", "coagulopathy", "obesity", "weight_loss",
    "fluid_electrolyte_disorders", "blood_loss_anemia", "deficiency_anemia",
    "alcohol_abuse", "drug_abuse", "psychoses", "depression",
    "total_elixhauser_groups_per_record",
]

REQUIRED_FIELDS = ["no_group", "no", "group", "name"]


def build_table(meta: MetaData) -> Table:
    # 全部用 String，你原本也是這樣（除了 primary key）
    cols = [
        Column("no_group", String(120), nullable=False, primary_key=True),
        Column("no", String(120), nullable=False),
        Column("group", String(10), nullable=False),
        Column("name", String(120), nullable=False),
    ]

    # 其餘欄位都 String(30) or String(120)（照你原本）
    # 這裡直接依 TITLE 生成，避免手打一堆欄位出錯
    existing = {c.name for c in cols}
    for key in TITLE:
        if key in existing:
            continue
        # 你原本 ctno/csn/name/no/no_group 比較長，其餘大多 30
        if key in ("ctno", "csn"):
            cols.append(Column(key, String(30)))
        else:
            cols.append(Column(key, String(30)))

    return Table(TABLE_NAME, meta, *cols)


def normalize_row(row: dict) -> dict:
    """
    DictReader 會用 header 當 key。
    這裡保證：
    - TITLE 裡每個 key 都存在（缺的補空字串）
    - 去掉 None
    - strip 空白
    """
    out = {}
    for k in TITLE:
        v = row.get(k, "")
        if v is None:
            v = ""
        if isinstance(v, str):
            v = v.strip()
        out[k] = v
    return out


def main():
    db_url = DEFAULT_DB_URL
    csv_path = Path(DEFAULT_CSV_PATH)

    # 支援 CLI：python pji_csv_to_db.py path/to.csv
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path.resolve()}")
        print("用法：python scripts/pji_csv_to_db.py /path/to/Revision_PJI_test_2.csv")
        sys.exit(1)

    engine = create_engine(db_url, echo=ECHO_SQL, future=True)
    meta = MetaData()

    table = build_table(meta)
    meta.create_all(engine)

    inserted = 0
    skipped = 0
    dup = 0

    with engine.begin() as conn:
        if TRUNCATE_BEFORE_IMPORT:
            conn.execute(text(f"TRUNCATE TABLE {TABLE_NAME}"))

        # 用 utf-8-sig：有些 Excel 會帶 BOM
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            # 如果 CSV header 不是 TITLE（例如順序/名稱不同），直接報錯讓你知道
            missing_headers = [h for h in REQUIRED_FIELDS if h not in (reader.fieldnames or [])]
            if missing_headers:
                print("[ERROR] CSV header 缺少必要欄位：", missing_headers)
                print("你 CSV 的 header：", reader.fieldnames)
                sys.exit(2)

            for i, row in enumerate(reader, start=2):  # 第2行開始是資料（第1行 header）
                # 空白行：DictReader 會給你一堆 None 或空字串
                if row is None or all((v is None or str(v).strip() == "") for v in row.values()):
                    skipped += 1
                    continue

                obj = normalize_row(row)

                # 必填欄位檢查
                if any(not obj.get(k) for k in REQUIRED_FIELDS):
                    print(f"[SKIP] line {i}: required fields missing -> "
                          f"{ {k: obj.get(k) for k in REQUIRED_FIELDS} }")
                    skipped += 1
                    continue

                try:
                    conn.execute(table.insert().values(**obj))
                    inserted += 1
                except IntegrityError as e:
                    # primary key 重複
                    dup += 1
                    print(f"[DUP] line {i}: no_group={obj.get('no_group')} -> {e.orig}")
                except Exception as e:
                    print(f"[ERROR] line {i}: {e}")
                    skipped += 1

    print("\n=== DONE ===")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Inserted: {inserted}")
    print(f"Duplicates: {dup}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
