# app.py
import os
import json
import time
import logging
import csv
import glob
from contextlib import contextmanager

import psutil
import pymysql
from pymysql.cursors import DictCursor

from datetime import timedelta

from flask import (
    Flask, render_template, request, redirect,
    session, make_response, flash, jsonify
)
from flask_bootstrap import Bootstrap5
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from apscheduler.schedulers.background import BackgroundScheduler

from db import db
from model import User


DEFAULT_DATABASE_URL = "mysql+pymysql://root@127.0.0.1:3306/PJI"
DEFAULT_SESSION_MINUTES = 10
DEFAULT_APP_HOST = "127.0.0.1"
DEFAULT_APP_PORT = 5001
DEFAULT_APP_DEBUG = True
DECISION_RULE_DIR = "Decision_rule"
INDEX_ACTION_ROUTES = {
    "0": "/personal_info?p_id={patient_id}",
    "1": "/model_diagnosis?p_id={patient_id}",
    "2": "/reactive_diagram?p_id={patient_id}",
    "3": "/upload_new_data?p_id={patient_id}",
    "5": "/pick_new_data?p_id={patient_id}",
    "6": "/pick_new_data_view?p_id={patient_id}",
    "7": "/back_new_data?p_id={patient_id}",
    "8": "/merge_new_data",
}


# -------------------------
# logging
# -------------------------
logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# -------------------------
# Lazy import helpers (重點：用到才 import)
# -------------------------
def lazy_import_personal_dp2():
    import personal_DecisionPath2
    return personal_DecisionPath2


def lazy_import_reactive_dp():
    import personal_DecisionPath_for_reactive
    return personal_DecisionPath_for_reactive


def lazy_import_vision_compare():
    import Vision_compare
    return Vision_compare


# -------------------------
# App factory
# -------------------------
def create_app():
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24))
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # ✅ 優先吃環境變數 DATABASE_URL，沒有就用本機
    # 例：mysql+pymysql://root@127.0.0.1:3306/PJI
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", DEFAULT_DATABASE_URL
    )

    app.config["WTF_CSRF_ENABLED"] = False
    app.permanent_session_lifetime = timedelta(
        minutes=int(os.environ.get("SESSION_MINUTES", str(DEFAULT_SESSION_MINUTES)))
    )

    Bootstrap5(app)
    db.init_app(app)
    Migrate(app, db)

    return app


app = create_app()

# ✅ SocketIO 強制用 threading（避免 eventlet greendns 慢）
socketio = SocketIO(app, async_mode="threading")

# ✅ Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20000 per day", "1000 per hour"],
    storage_uri="memory://",
)

# -------------------------
# Dynamic rate limit (scheduler)
# -------------------------
current_limit = "5 per minute"
_scheduler = None


def adjust_rate_limiting():
    """注意：不要 interval=1，會硬卡 1 秒"""
    global current_limit

    cpu_usage = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()

    # 你可以自己調整門檻
    if cpu_usage > 75 or memory.percent > 80:
        current_limit = "10 per minute"
    else:
        current_limit = "500 per minute"

    logging.info("CPU=%s MEM=%s current_limit=%s", cpu_usage, memory.percent, current_limit)
    return current_limit


def start_scheduler_once():
    """
    Flask debug reloader 會跑兩次：
      - parent process (WERKZEUG_RUN_MAIN not set)
      - child process  (WERKZEUG_RUN_MAIN="true")

    ✅ 只在真正 serving 的 child process 啟動 scheduler
    ✅ 非 debug 模式就直接啟動
    """
    global _scheduler
    if _scheduler is not None:
        return

    is_debug = bool(app.debug)
    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    if is_debug and not is_reloader_child:
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(func=adjust_rate_limiting, trigger="interval", seconds=5)
    _scheduler.start()
    logging.info("Scheduler started. debug=%s child=%s", is_debug, is_reloader_child)


# -------------------------
# Error handler
# -------------------------
@app.errorhandler(429)
def rate_limit_exceeded(e):
    response = make_response(
        '<html><body><h1 style="font-size: 32px;">系統繁忙，請稍後再試。</h1></body></html>',
        429,
    )
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


@app.route("/adjust_rate_limiting", methods=["GET", "POST"])
def adjust_rate_limiting_route():
    return str(adjust_rate_limiting())


# -------------------------
# MySQL connect helper (統一管理帳密)
# -------------------------
def mysql_conn(db_name="PJI"):
    """
    用環境變數控管比較安全：
      MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_DB
    如果你 root 沒密碼，MYSQL_PASSWORD 留空即可。
    """
    host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    dbn = os.environ.get("MYSQL_DB", db_name)

    return pymysql.connect(host=host, user=user, password=password, port=port, db=dbn)


@contextmanager
def mysql_cursor(db_name="PJI"):
    conn = mysql_conn(db_name)
    cur = conn.cursor()
    try:
        yield conn, cur
    finally:
        conn.close()


def db_fetch_all(sql, params=None, db_name="PJI"):
    with mysql_cursor(db_name) as (_conn, cur):
        cur.execute(sql, params or ())
        return cur.fetchall()


def db_execute(sql, params=None, commit=True, db_name="PJI"):
    with mysql_cursor(db_name) as (conn, cur):
        cur.execute(sql, params or ())
        if commit:
            conn.commit()


def get_current_user_or_redirect():
    session.permanent = True
    uid = session.get("user-id")
    if uid is None:
        return None, redirect("/auth_login")

    user = User.get_by_uid(uid)
    if user is None:
        return None, redirect("/auth_login")

    return user, None


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json_file(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


def load_json_file_safe(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logging.warning("load_json_file_safe failed for %s: %s", path, exc)
        return default


def read_patient_rows(name):
    sql = "SELECT * FROM PJI.revision_pji WHERE no_group = %s"
    return db_fetch_all(sql, (name,), db_name="PJI")


def read_patient_row_dict(name):
    conn = mysql_conn("PJI")
    try:
        cur = conn.cursor(DictCursor)
        cur.execute("SELECT * FROM PJI.revision_pji WHERE no_group = %s LIMIT 1", (name,))
        return cur.fetchone()
    finally:
        conn.close()


def read_rows_from_table(table_name):
    try:
        sql = f"SELECT * FROM PJI.{table_name} ORDER BY {table_name}.no_group;"
        return db_fetch_all(sql, db_name="PJI")
    except Exception as exc:
        logging.warning("read_rows_from_table failed for %s: %s", table_name, exc)
        return []


def ensure_message_table():
    sql = """
        CREATE TABLE IF NOT EXISTS PJI.message (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    db_execute(sql, db_name="PJI")


def ensure_new_data_tables():
    with mysql_cursor("PJI") as (conn, cur):
        cur.execute("CREATE TABLE IF NOT EXISTS PJI.pji_new_data LIKE PJI.revision_pji")
        cur.execute("CREATE TABLE IF NOT EXISTS PJI.pji_new_data_buffer LIKE PJI.revision_pji")
        cur.execute("SELECT COUNT(*) FROM PJI.pji_new_data")
        new_data_count = cur.fetchone()[0]
        if new_data_count == 0:
            cur.execute("INSERT INTO PJI.pji_new_data SELECT * FROM PJI.revision_pji")
        conn.commit()


def import_revision_csv(file_path):
    """
    匯入 CSV 到 revision_pji / pji_new_data（upsert）。
    只填必要欄位，避免 schema 約束造成整批失敗。
    """
    inserted = 0
    updated = 0
    with mysql_cursor("PJI") as (conn, cur):
        ensure_new_data_tables()
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                no_group = (row.get("no_group") or "").strip()
                no = (row.get("no") or "").strip()
                group = (row.get("group") or "").strip()
                name = (row.get("name") or "").strip()
                if not (no_group and no and group and name):
                    continue

                computed = f"CT-{no_group}"
                sql = (
                    "INSERT INTO revision_pji (no_group, no, `group`, name, computed_tomography_number) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "no = VALUES(no), `group` = VALUES(`group`), "
                    "name = VALUES(name), computed_tomography_number = VALUES(computed_tomography_number)"
                )
                cur.execute(sql, (no_group, no, group, name, computed))
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1

                cur.execute(
                    "INSERT INTO pji_new_data (no_group, no, `group`, name, computed_tomography_number) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "no = VALUES(no), `group` = VALUES(`group`), "
                    "name = VALUES(name), computed_tomography_number = VALUES(computed_tomography_number)",
                    (no_group, no, group, name, computed),
                )
        conn.commit()
    return inserted, updated


def first_json_match(pattern, default):
    candidates = sorted(glob.glob(pattern))
    for path in candidates:
        payload = load_json_file_safe(path, None)
        if payload:
            return payload
    return default


def normalize_patient_key(name):
    return "".join(c for c in str(name) if c.isalnum())


def build_default_decision_rule_map():
    return {
        "A": "Age >= 65",
        "B": "Segment (%) >= 70",
        "C": "HGB <= 12",
        "D": "PLATELET >= 250",
        "E": "Serum WBC >= 10",
        "F": "P.T >= 12",
        "G": "APTT >= 30",
        "H": "Total CCI >= 2",
        "I": "Total Elixhauser >= 1",
        "J": "Revision surgery",
        "K": "ASA >= 2",
        "L": "2X positive culture",
        "M": "Serum CRP >= 10",
        "N": "Serum ESR >= 30",
        "O": "Synovial WBC >= 3000",
        "P": "Single Positive culture",
        "Q": "Synovial PMN >= 70",
        "R": "Positive Histology",
        "S": "Purulence",
    }


def build_model_diagnosis_payload(name):
    patient_key = normalize_patient_key(name)

    decision_list_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/decision_rule_{patient_key}.json", None
    )
    if not decision_list_json:
        decision_list_json = load_json_file_safe(
            f"{DECISION_RULE_DIR}/decision_rule.json", {"0": ["A"], "1": ["I", "C"]}
        )

    rule_map_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/decision_rule_map_{patient_key}.json", None
    )
    if not rule_map_json:
        rule_map_json = first_json_match(
            f"{DECISION_RULE_DIR}/decision_rule_map_*.json",
            build_default_decision_rule_map(),
        )

    simplified_rule_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/simplified_decision_rule_{patient_key}.json", None
    )
    if not simplified_rule_json:
        simplified_rule_json = first_json_match(
            f"{DECISION_RULE_DIR}/simplified_decision_rule_*.json",
            decision_list_json,
        )

    simplified_rule_map_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/simplified_decision_rule_map_{patient_key}.json", None
    )
    if not simplified_rule_map_json:
        simplified_rule_map_json = first_json_match(
            f"{DECISION_RULE_DIR}/simplified_decision_rule_map_*.json",
            rule_map_json,
        )

    if not isinstance(decision_list_json, dict):
        decision_list_json = {"0": ["A"]}
    if not isinstance(rule_map_json, dict):
        rule_map_json = build_default_decision_rule_map()
    if not isinstance(simplified_rule_json, dict):
        simplified_rule_json = decision_list_json
    if not isinstance(simplified_rule_map_json, dict):
        simplified_rule_map_json = rule_map_json

    return {
        "decision_list_json": decision_list_json,
        "rule_map_json": rule_map_json,
        "simplified_rule_json": simplified_rule_json,
        "simplified_rule_map_json": simplified_rule_map_json,
        "decision_path_num": max(1, len(decision_list_json)),
    }


def to_float_or_zero(value):
    try:
        if value is None or str(value).strip() == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def to_binary_flag(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "positive", "infected", "chronic"}:
        return 1.0
    try:
        return 1.0 if float(text) >= 1 else 0.0
    except Exception:
        return 0.0


def to_asa2_flag(value):
    try:
        return 1.0 if float(value) >= 2 else 0.0
    except Exception:
        return 0.0


def build_predict_arr_from_patient_dict(row):
    if not row:
        return [0.0] * 19
    return [
        to_float_or_zero(row.get("age")),
        to_float_or_zero(row.get("segment")),
        to_float_or_zero(row.get("hgb")),
        to_float_or_zero(row.get("platelet")),
        to_float_or_zero(row.get("serum_WBC")),
        to_float_or_zero(row.get("p_t")),
        to_float_or_zero(row.get("aptt")),
        to_float_or_zero(row.get("total_cci")),
        to_float_or_zero(row.get("total_elixhauser_groups_per_record")),
        to_binary_flag(row.get("primary_revision_native_hip")),
        to_asa2_flag(row.get("asa")),
        to_binary_flag(row.get("positive_culture")),
        to_float_or_zero(row.get("Serum_CRP")),
        to_float_or_zero(row.get("serum_ESR")),
        to_float_or_zero(row.get("synovial_WBC")),
        to_binary_flag(row.get("single_positive_culture")),
        to_float_or_zero(row.get("synovial_Neutrophil")),
        to_binary_flag(row.get("positive_histology")),
        to_binary_flag(row.get("pulurence")),
    ]


def predict_personal_result_text(name):
    row = read_patient_row_dict(name)
    arr = build_predict_arr_from_patient_dict(row)

    try:
        Vision_compare = lazy_import_vision_compare()
        predict_data = Vision_compare.tran_df(arr)
        result = Vision_compare.stacking_predict(predict_data)
        return "Infected!" if result[0] == 1 else "Aseptic!"
    except Exception as exc:
        logging.warning("predict_personal_result_text fallback for %s: %s", name, exc)
        if arr[12] >= 10 or arr[13] >= 30 or arr[14] >= 3000 or arr[16] >= 70:
            return "Infected!"
        return "Aseptic!"


def build_reactive_arr_from_row(row):
    if not row:
        return [0.0] * 19
    idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    return [to_float_or_zero(row[i] if i < len(row) else 0) for i in idx]


def build_reactive_payload(name, user_data):
    reactive_rule_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/reactive_rule.json", {"0": ["A"], "1": ["B"]}
    )
    reactive_rule_map_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/reactive_rule_map.json", build_default_decision_rule_map()
    )
    reactive_decision_list_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/decision_rule_reactive_diagram.json", {"0": ["A", "N", "Q"]}
    )
    reactive_decision_list_map_json = load_json_file_safe(
        f"{DECISION_RULE_DIR}/decision_rule_reactive_diagram_map.json",
        build_default_decision_rule_map(),
    )

    arr = load_json_file_safe(f"{DECISION_RULE_DIR}/reactived_data_onlyvalue.json", None)
    if not isinstance(arr, list) or len(arr) < 19:
        arr = build_reactive_arr_from_row(user_data[0] if user_data else None)
        save_json_file(f"{DECISION_RULE_DIR}/reactived_data_onlyvalue.json", arr)

    reactived_data_json = load_json_file_safe(f"{DECISION_RULE_DIR}/reactived_data.json", None)
    if not isinstance(reactived_data_json, dict):
        keys = list("ABCDEFGHIJKLMNOPQRS")
        reactived_data_json = {k: v for k, v in zip(keys, arr)}
        save_json_file(f"{DECISION_RULE_DIR}/reactived_data.json", reactived_data_json)

    result_text = "Aseptic"
    result_xgb = "N/A"
    result_rf = "N/A"
    result_nb = "N/A"
    result_lr = "N/A"

    try:
        Vision_compare = lazy_import_vision_compare()
        predict_data = Vision_compare.tran_df(arr)
        result1 = Vision_compare.stacking_predict(predict_data)
        result_text = "Infected" if result1[0] == 1 else "Aseptic"
        result_xgb = str(Vision_compare.xgboost_predict(predict_data)[0])
        result_rf = str(Vision_compare.rf_predict(predict_data)[0])
        result_nb = str(Vision_compare.nb_predict(predict_data)[0])
        result_lr = str(Vision_compare.lr_predict(predict_data)[0])
    except Exception as exc:
        logging.warning("Vision_compare predict failed, use fallback: %s", exc)
        if arr[12] >= 10 or arr[13] >= 30 or arr[14] >= 3000 or arr[16] >= 70:
            result_text = "Infected"
            result_xgb = result_rf = result_nb = result_lr = "1"
        else:
            result_text = "Aseptic"
            result_xgb = result_rf = result_nb = result_lr = "0"

    return {
        "result": result_text,
        "result_xgb": result_xgb,
        "result_rf": result_rf,
        "result_nb": result_nb,
        "result_lr": result_lr,
        "predict_data": arr,
        "reactive_rule_json": reactive_rule_json,
        "reactive_rule_map_json": reactive_rule_map_json,
        "reactive_decision_list_json": reactive_decision_list_json,
        "reactive_decision_list_map_json": reactive_decision_list_map_json,
        "reactived_data_json": reactived_data_json,
    }


def str_to_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_index_action(btn_id, patient_id):
    route = INDEX_ACTION_ROUTES.get(btn_id, "/")
    if "{patient_id}" in route:
        return route.format(patient_id=patient_id)
    return route


def get_new_data_row_or_redirect(name, source_table, fallback_route):
    rows = read_rows_with_name(source_table, name)
    if not rows:
        return None, redirect(fallback_route)
    return rows[0], None


# -------------------------
# Routes
# -------------------------
@app.route("/auth_login", methods=["GET", "POST"])
@limiter.limit(lambda: current_limit)
def auth_login():
    if request.method == "GET":
        return render_template("auth_login.html")

    username = request.form.get("username", "")
    password = request.form.get("password", "")

    user = User.get_by_name(username)
    if user and user.check_password(password):
        session_hash = username + str(os.urandom(30))
        user.session_hash = session_hash
        db.session.commit()
        session["user-id"] = session_hash
        return redirect("/")

    return redirect("/auth_login")


@app.route("/", methods=["GET", "POST"])
def index():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        btn_id = str(payload.get("btnID", ""))
        patient_id = str(payload.get("patientID", "")).strip()
        if not patient_id:
            return "/", 400
        return resolve_index_action(btn_id, patient_id)

    rows = read_rows_from_table("revision_pji")

    return render_template("index.html", u=rows, name=user)


@app.route("/model_diagnosis")
def model_diagnosis():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    name = request.args.get("p_id", "").strip()
    if not name:
        return "Invalid p_id", 400

    payload = build_model_diagnosis_payload(name)
    return render_template("model_diagnosis.html", name=name, username=user, **payload)


@app.route("/train_new_data", methods=["GET", "POST"])
def train_new_data():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    rows = read_rows_from_table("pji_new_data")
    return render_template("train_new_data.html", u=rows, name=user)


@app.route("/new_data_buffer", methods=["GET", "POST"])
def new_data_buffer():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    rows = read_rows_from_table("pji_new_data_buffer")
    return render_template("new_data_buffer.html", u=rows, name=user)


@app.route("/upload_new_data_csv", methods=["GET", "POST"])
def upload_new_data_csv():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    if request.method == "POST":
        ensure_new_data_tables()
        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"message": "No file uploaded"}), 400

        lower_name = uploaded_file.filename.lower()
        if not (lower_name.endswith(".csv") or lower_name.endswith(".xlsx")):
            return jsonify({"message": "Please upload a CSV or Excel file."}), 400

        os.makedirs("uploads", exist_ok=True)
        extension = ".csv" if lower_name.endswith(".csv") else ".xlsx"
        file_path = os.path.join("uploads", f"New_data{extension}")
        uploaded_file.save(file_path)
        csv_path = file_path

        if extension == ".xlsx":
            try:
                import pandas as pd

                csv_path = os.path.join("uploads", "New_data_from_excel.csv")
                pd.read_excel(file_path).to_csv(csv_path, index=False)
            except Exception as exc:
                logging.exception("xlsx parse failed: %s", exc)
                return jsonify({"message": "Failed to parse Excel file."}), 400

        try:
            import_revision_csv(csv_path)
        except Exception as exc:
            logging.exception("import_revision_csv failed: %s", exc)
            return jsonify({"message": "Failed to import uploaded data into DB."}), 500

        for i in range(1, 101):
            if i <= 30:
                time.sleep(0.02)
            socketio.emit("progress_percent", {"percentage": i})
        socketio.emit("task_completed")
        return "File uploaded successfully."

    return render_template("upload_new_data_csv.html", name=user)


@app.route("/message_board", methods=["GET", "POST"])
def message_board():
    _user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        content = request.form.get("content", "").strip()
        if not username or not content:
            flash("Username and message are required.", "error")
            return render_template("message_board.html")

        try:
            ensure_message_table()
            db_execute(
                "INSERT INTO PJI.message (username, content) VALUES (%s, %s)",
                (username, content),
                db_name="PJI",
            )
            flash("Leave a message successfully!", "success")
        except Exception as exc:
            logging.exception("message_board insert failed: %s", exc)
            flash("Failed to leave a message.", "error")

    return render_template("message_board.html")


@app.route("/logout")
def logout():
    session["user-id"] = None
    return redirect("/auth_login")


@app.route("/personal_info")
@limiter.limit(lambda: current_limit)
def personal_info():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    name = request.args.get("p_id", "").strip()
    if not name:
        return "Invalid p_id", 400

    rows = read_patient_rows(name)
    if not rows:
        return "Patient not found", 404

    result_text = predict_personal_result_text(name)

    return render_template(
        "personal_info.html",
        name=name,
        u=rows,
        result=result_text,
        username=user,
    )


def move_patient_between_tables(name, src_table, dst_table):
    allowed_tables = {"pji_new_data", "pji_new_data_buffer"}
    if src_table not in allowed_tables or dst_table not in allowed_tables:
        raise ValueError("Invalid table name for move operation")

    with mysql_cursor("PJI") as (conn, cur):
        # Keep move idempotent: remove any stale duplicate in destination first.
        cur.execute(f"DELETE FROM PJI.{dst_table} WHERE no_group = %s", (name,))
        cur.execute(
            f"INSERT INTO PJI.{dst_table} SELECT * FROM PJI.{src_table} WHERE no_group = %s",
            (name,),
        )
        cur.execute(f"DELETE FROM PJI.{src_table} WHERE no_group = %s", (name,))
        conn.commit()


@app.route("/upload_new_data")
def upload_new_data():
    _user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    name = request.args.get("p_id", "").strip()
    target_route = "/new_data_buffer"
    if not name:
        target_route = "/train_new_data"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return target_route
        return redirect(target_route)

    try:
        move_patient_between_tables(name, "pji_new_data", "pji_new_data_buffer")
    except Exception as exc:
        logging.warning("upload_new_data move failed for %s: %s", name, exc)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return target_route
    return redirect(target_route)


@app.route("/back_new_data")
def back_new_data():
    _user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    name = request.args.get("p_id", "").strip()
    target_route = "/train_new_data"
    if not name:
        target_route = "/new_data_buffer"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return target_route
        return redirect(target_route)

    try:
        move_patient_between_tables(name, "pji_new_data_buffer", "pji_new_data")
    except Exception as exc:
        logging.warning("back_new_data move failed for %s: %s", name, exc)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return target_route
    return redirect(target_route)


@app.route("/merge_new_data")
def merge_new_data():
    _user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    with mysql_cursor("PJI") as (conn, cur):
        cur.execute("INSERT IGNORE INTO PJI.pji_new_data SELECT * FROM PJI.pji_new_data_buffer")
        cur.execute("DELETE FROM PJI.pji_new_data_buffer")
        conn.commit()
    return redirect("/train_new_data")


def infer_result_text_from_row(row):
    if not row:
        return "N/A"
    serum_crp = to_float_or_zero(row[13] if len(row) > 13 else 0)
    serum_esr = to_float_or_zero(row[14] if len(row) > 14 else 0)
    synovial_wbc = to_float_or_zero(row[15] if len(row) > 15 else 0)
    synovial_pmn = to_float_or_zero(row[17] if len(row) > 17 else 0)
    return "Infected" if (serum_crp >= 10 or serum_esr >= 30 or synovial_wbc >= 3000 or synovial_pmn >= 70) else "Aseptic"


@app.route("/pick_new_data")
def pick_new_data():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    name = request.args.get("p_id", "").strip()
    if not name:
        return redirect("/train_new_data")

    row, redirect_response = get_new_data_row_or_redirect(
        name, "pji_new_data", "/train_new_data"
    )
    if redirect_response:
        return redirect_response

    payload = build_model_diagnosis_payload(name)
    result_text = infer_result_text_from_row(row)
    return render_template(
        "pick_new_data.html",
        name=name,
        u=[row],
        result=result_text,
        username=user,
        decision_list=[],
        decision_list_json=payload["decision_list_json"],
        rule_map_json=payload["rule_map_json"],
    )


@app.route("/pick_new_data_view")
def pick_new_data_view():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    ensure_new_data_tables()
    name = request.args.get("p_id", "").strip()
    if not name:
        return redirect("/new_data_buffer")

    row, redirect_response = get_new_data_row_or_redirect(
        name, "pji_new_data_buffer", "/new_data_buffer"
    )
    if redirect_response:
        return redirect_response

    payload = build_model_diagnosis_payload(name)
    result_text = infer_result_text_from_row(row)
    return render_template(
        "pick_new_data_view.html",
        name=name,
        u=[row],
        result=result_text,
        username=user,
        decision_list=[],
        decision_list_json=payload["decision_list_json"],
        rule_map_json=payload["rule_map_json"],
    )


def read_rows_with_name(table_name, name):
    try:
        sql = f"SELECT * FROM PJI.{table_name} WHERE no_group = %s"
        return db_fetch_all(sql, (name,), db_name="PJI")
    except Exception as exc:
        logging.warning("read_rows_with_name failed for %s/%s: %s", table_name, name, exc)
        return []


@app.route("/reactive_diagram", methods=["GET", "POST"])
def reactive_diagram():
    user, redirect_response = get_current_user_or_redirect()
    if redirect_response:
        return redirect_response

    name = request.args.get("p_id", "").strip()
    if not name:
        return "Invalid p_id", 400

    user_data = read_patient_rows(name)
    if not user_data:
        return "Patient not found", 404

    os.makedirs(DECISION_RULE_DIR, exist_ok=True)
    save_json_file(f"{DECISION_RULE_DIR}/user_data.json", user_data)

    user_data_json = load_json_file(f"{DECISION_RULE_DIR}/user_data.json")

    if request.method == "GET":
        payload = build_reactive_payload(name, user_data)
        return render_template(
            "reactive_diagram.html",
            user=user,
            user_data_json=user_data_json,
            name=name,
            **payload,
        )

    return render_template("reactive_diagram.html", user=user, user_data_json=user_data_json, name=name)


# -------------------------
# SocketIO event
# -------------------------
@socketio.on("run_task")
def run_task(message):
    Vision_compare = lazy_import_vision_compare()
    personal_DecisionPath_for_reactive = lazy_import_reactive_dp()

    arr = message.get("arr", [])
    name = str(message.get("name", ""))

    arr = [float(x) for x in arr]

    os.makedirs(DECISION_RULE_DIR, exist_ok=True)

    save_json_file(f"{DECISION_RULE_DIR}/reactived_data_onlyvalue.json", arr)

    reactived_data_key = list("ABCDEFGHIJKLMNOPQRS")  # A~S 共 19 個
    reactived_data_dict = {k: v for k, v in zip(reactived_data_key, arr)}

    save_json_file(f"{DECISION_RULE_DIR}/reactived_data.json", reactived_data_dict)

    for i in range(1, 21):
        time.sleep(0.3)
        socketio.emit("task_progress", {"progress": i})

    predict_data = Vision_compare.tran_df(arr)
    try:
        patient_id = int(name)
        personal_DecisionPath_for_reactive.run_test(patient_id, predict_data)
    except Exception as exc:
        logging.warning("run_task skip run_test for patient %s: %s", name, exc)

    for i in range(21, 101):
        time.sleep(0.05)
        socketio.emit("task_progress", {"progress": i})

    socketio.emit("update_frontend", {"progress": 100})


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # ✅ 只在真正 serving 的程序啟 scheduler（debug reloader 不會跑兩次）
    start_scheduler_once()

    # debug=True 會開 reloader；scheduler 仍只會在 child 啟動
    app_host = os.environ.get("APP_HOST", DEFAULT_APP_HOST)
    app_port = int(os.environ.get("APP_PORT", str(DEFAULT_APP_PORT)))
    app_debug = str_to_bool(os.environ.get("APP_DEBUG"), default=DEFAULT_APP_DEBUG)

    socketio.run(
        app,
        host=app_host,
        port=app_port,
        debug=app_debug,
        allow_unsafe_werkzeug=True,
    )
