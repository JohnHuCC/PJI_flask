import vonage
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import joblib
import pandas as pd
import numpy as np
import Vision_compare
from flask import Flask, render_template, request, url_for, redirect, flash, session
import json
from db import db
from flask_bootstrap import Bootstrap5
from flask_migrate import Migrate
from model import User, RevisionPJI
import os
from datetime import timedelta
import pymysql
import personal_DecisionPath2
import personal_DecisionPath_for_reactive
from flask import jsonify, request, make_response
from flask_socketio import SocketIO
import time
import csv
import html
from urllib.parse import urlparse
from preprocessing_data import preprocessing_data, upload_to_db
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
import logging

logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)


app.config['SECRET_KEY'] = os.urandom(24)
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=31)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:love29338615@127.0.0.1:3306/PJI"
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')


app.config['WTF_CSRF_ENABLED'] = False
app.permanent_session_lifetime = timedelta(minutes=10)
bootstrap = Bootstrap5(app)
db.init_app(app)
migrate = Migrate(app, db)

reactived_data_key = ["A", "B", "C", "D", "E", "F", "G",
                      "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]


socketio = SocketIO(app)


limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20000 per day", "1000 per hour"],
    storage_uri="memory://",
)

# 自定義錯誤處理器


@app.errorhandler(429)
def rate_limit_exceeded(e):
    response = make_response(
        '<html><body><h1 style="font-size: 32px;">系統繁忙，請稍後再試。</h1></body></html>', 429)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


global current_limit
current_limit = "5 per minute"  # 初始速率限制


@app.route("/adjust_rate_limiting", methods=['GET', 'POST'])
def adjust_rate_limiting():
    global current_limit
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory.percent}%")
    if cpu_usage > 75 or memory.percent > 80:
        current_limit = "10 per minute"
    else:
        current_limit = "500 per minute"
    print('current_limit:', current_limit)
    return current_limit


scheduler = BackgroundScheduler()
scheduler.add_job(func=adjust_rate_limiting,
                  trigger="interval", seconds=5)
scheduler.start()


@app.route("/auth_login", methods=['GET', 'POST'])
@limiter.limit(lambda: current_limit)
def auth_login():
    if request.method == 'GET':
        return render_template('auth_login.html')
    elif request.method == 'POST':
        print('db:', db)
        username = request.form['username']
        user = User.get_by_name(username)
        if user.check_password(request.form['password']):
            session_hash = username + str(os.urandom(30))
            user.session_hash = session_hash
            db.session.commit()
            session['user-id'] = session_hash
            return redirect('/')
        else:
            return redirect('/auth_login')


@socketio.on('run_task')
def run_task(message):
    arr = message.get('arr', [])
    arr[:] = [float(x) for x in arr]
    with open("Decision_rule/reactived_data_onlyvalue.json", "w") as file:
        json.dump(arr, file)
    name = message.get('name', "")
    print("calling reactived_data_task")
    global result1
    reactived_data_key = ["A", "B", "C", "D", "E", "F", "G",
                          "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]
    print('success reactived_data_key:', reactived_data_key)
    reactived_data_dict = {k: v for k, v in zip(reactived_data_key, arr)}

    with open("Decision_rule/reactived_data.json", "w") as file:
        json.dump(reactived_data_dict, file)
    for i in range(1, 21):
        time.sleep(0.3)
        progress = i
        socketio.emit('task_progress', {'progress': progress})
    for i in range(21, 36):
        time.sleep(0.5)
        progress = i
        socketio.emit('task_progress', {'progress': progress})
    predict_data = Vision_compare.tran_df(arr)
    reactive_diagram_dp = personal_DecisionPath_for_reactive.run_test(
        int(name), predict_data)
    for i in range(36, 100):
        time.sleep(0.1)
        progress = i
        socketio.emit('task_progress', {'progress': progress})
    socketio.emit('task_progress', {'progress': 100})
    socketio.emit('update_frontend', {'progress': 100})


def tran_df(arr):
    predict_data = pd.DataFrame(
        np.array([arr]), columns=['Age', 'Segment', 'HGB', 'PLATELET', 'Serum ', 'P.T', 'APTT', 'Total CCI', 'Total Elixhauser Groups per record', 'Primary,Revision,native hip', 'ASA_2',
                                  '2X positive culture', 'Serum CRP', 'Serum ESR', 'Synovial WBC', 'Single Positive culture', 'Synovial_PMN', 'Positive Histology', 'Purulence'
                                  ])
    return predict_data


@ app.route('/', methods=['GET', 'POST'])
def index():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')

    if uid != None:
        user = User.get_by_uid(uid)
        print(user)
        if user == None:
            # no such user
            redirect('/auth_login')
    else:
        redirect('/auth_login')
    # 获取 URI 信息
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    print('db_uri_info:', db_uri_info)
    # conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
    #                        password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.revision_pji ORDER BY CAST(revision_pji.no_group AS unsigned);"
    cur.execute(sql)
    u = cur.fetchall()
    conn.close()

    if request.method == 'POST':
        dict_p = request.get_json()
        if dict_p["btnID"] == "0":
            p_id = dict_p["patientID"]
            return url_for('personal_info', p_id=p_id)
        elif dict_p["btnID"] == "1":
            p_id = dict_p["patientID"]
            return url_for('model_diagnosis', p_id=p_id)
        elif dict_p["btnID"] == "2":
            p_id = dict_p["patientID"]
            return url_for('reactive_diagram', p_id=p_id)
        elif dict_p["btnID"] == "3":
            p_id = dict_p["patientID"]
            return url_for('upload_new_data', p_id=p_id)
        elif dict_p["btnID"] == "5":
            p_id = dict_p["patientID"]
            return url_for('pick_new_data', p_id=p_id)
        elif dict_p["btnID"] == "6":
            p_id = dict_p["patientID"]
            return url_for('pick_new_data_view', p_id=p_id)
        elif dict_p["btnID"] == "7":
            p_id = dict_p["patientID"]
            return url_for('back_new_data', p_id=p_id)

    return render_template('index.html', u=u, name=user)


@ app.route('/new_data_buffer', methods=['GET', 'POST'])
def new_data_buffer():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')

    if uid != None:
        user = User.get_by_uid(uid)
        print(user)
        if user == None:
            # no such user
            redirect('/auth_login')
    else:
        redirect('/auth_login')

    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    # conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
    #                        password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.pji_new_data_buffer ORDER BY CAST(pji_new_data_buffer.no_group AS unsigned);"
    cur.execute(sql)
    u = cur.fetchall()
    conn.close()

    # if request.method == 'POST':
    #     dict_p = request.get_json()
    #     if dict_p["btnID"] == "7":
    #         p_id = dict_p["patientID"]
    #         return url_for('train_new_data', p_id=p_id)

    return render_template('new_data_buffer.html', u=u, name=user)


@ app.route('/train_new_data', methods=['GET', 'POST'])
def train_new_data():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')

    if uid != None:
        user = User.get_by_uid(uid)
        print(user)
        if user == None:
            # no such user
            redirect('/auth_login')
    else:
        redirect('/auth_login')

    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
                           password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.pji_new_data ORDER BY CAST(pji_new_data.no_group AS unsigned);"
    cur.execute(sql)
    u = cur.fetchall()
    conn.close()

    if request.method == 'POST':
        dict_p = request.get_json()
        if dict_p["btnID"] == "5":
            p_id = dict_p["patientID"]
            return url_for('pick_new_data', p_id=p_id)

    return render_template('train_new_data.html', u=u, name=user)


@ app.route("/auth_register", methods=['GET', 'POST'])
def auth_register():
    if request.method == 'GET':
        return render_template('auth_register.html')
    elif request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.get_by_name(username) == None:
            User.insert(username, password)
            flash('user created')
            return redirect('/auth_login')
        else:
            flash('user existed')
            return redirect('/auth_register')


@ app.route('/logout')
def logout():
    session['user-id'] = False
    # flash('Log Out See You.')
    return redirect('/auth_login')


@ app.route("/auth_password")
def auth_password():
    return render_template('auth_password.html')


personal_result_view = None


@ app.route("/pick_new_data_view")
def pick_new_data_view():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')
    # uid = session['user-id']
    if uid != None:
        user = User.get_by_uid(uid)
        if user == None:
            redirect('/auth_login')
        else:
            username = user
    name = request.args.get('p_id')
    decision_result_json = json.load(
        open("decision_result_"+name+".json"))
    if (decision_result_json["result"] == "1"):
        result_text = "Infected"
    elif (decision_result_json["result"] == "0"):
        result_text = "Aseptic"
    else:
        result_text = "None"
    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
                           password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.pji_new_data_buffer WHERE no_group =" + str(name)
    cur.execute(sql)
    new_data = cur.fetchall()
    conn.close()

    decision_list_file = open("decision_list_file.txt", "r")
    decision_list_file_content = decision_list_file.read().strip()
    decision_list = decision_list_file_content.split("\n")
    decision_list_file.close()
    decision_list_json = json.load(
        open("decision_rule_"+name+".json"))
    rule_map_json = json.load(
        open("decision_rule_map_"+name+".json"))
    return render_template('pick_new_data_view.html', name=name, u=new_data, result=result_text, username=username, decision_list=decision_list, decision_list_json=decision_list_json, rule_map_json=rule_map_json)


@ app.route("/pick_new_data")
def pick_new_data():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')
    # uid = session['user-id']
    if uid != None:
        user = User.get_by_uid(uid)
        if user == None:
            redirect('/auth_login')
        else:
            username = user
    name = request.args.get('p_id')
    personal_result = smote.personalDP(int(name), 7)
    personal_result_view = personal_result
    if (personal_result == 1):
        result_text = "Infected"
    else:
        result_text = "Aseptic"
    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
                           password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.pji_new_data WHERE no_group =" + str(name)
    cur.execute(sql)
    new_data = cur.fetchall()
    conn.close()

    decision_list_file = open("decision_list_file.txt", "r")
    decision_list_file_content = decision_list_file.read().strip()
    decision_list = decision_list_file_content.split("\n")
    decision_list_file.close()
    decision_list_json = json.load(
        open("Decision_rule/decision_rule_"+name+".json"))
    rule_map_json = json.load(
        open("Decision_rule/decision_rule_map_"+name+".json"))
    return render_template('pick_new_data.html', name=name, u=new_data, result=result_text, username=username, decision_list=decision_list, decision_list_json=decision_list_json, rule_map_json=rule_map_json)


@ app.route("/upload_new_data", methods=['GET', 'POST'])
def upload_new_data():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')

    if uid != None:
        user = User.get_by_uid(uid)
        print(user)
        if user == None:
            # no such user
            redirect('/auth_login')
    else:
        redirect('/auth_login')
    name = request.args.get('p_id')
    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
                           password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql_delete = f"DELETE FROM PJI.pji_new_data WHERE no_group = {str(name)};"
    sql_insert = f"INSERT INTO PJI.pji_new_data_buffer SELECT * FROM PJI.pji_new_data WHERE no_group = {str(name)};"
    cur.execute(sql_insert)
    cur.execute(sql_delete)
    conn.commit()
    conn.close()
    return url_for('new_data_buffer')


@ app.route("/back_new_data", methods=['GET', 'POST'])
def back_new_data():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')

    if uid != None:
        user = User.get_by_uid(uid)
        print(user)
        if user == None:
            # no such user
            redirect('/auth_login')
    else:
        redirect('/auth_login')
    name = request.args.get('p_id')
    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
                           password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql_insert = f"INSERT INTO PJI.pji_new_data SELECT * FROM PJI.pji_new_data_buffer WHERE no_group = {str(name)};"
    sql_delete = f"DELETE FROM PJI.pji_new_data_buffer WHERE no_group = {str(name)};"
    cur.execute(sql_insert)
    cur.execute(sql_delete)
    conn.commit()
    conn.close()
    return redirect('train_new_data')


@ app.route("/chart3")
def chart3():
    return render_template('chart3.html')


@ app.route('/model_diagnosis', methods=['GET', 'POST'])
def model_diagnosis():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')
    name = request.args.get('p_id')
    user = User.get_by_uid(uid)
    # decision_list_file = open("decision_list_file.txt", "r")
    # decision_list_file_content = decision_list_file.read().strip()
    # decision_list = decision_list_file_content.split("\n")
    # decision_list_file.close()
    decision_list_json = json.load(
        open("Decision_rule/decision_rule_"+name+".json"))
    simplified_rule_json = json.load(
        open("Decision_rule/simplified_decision_rule_"+name+".json"))
    rule_map_json = json.load(
        open("Decision_rule/decision_rule_map_"+name+".json"))
    simplified_rule_map_json = json.load(
        open("Decision_rule/simplified_decision_rule_map_"+name+".json"))
    return render_template('model_diagnosis.html', decision_list_json=decision_list_json, rule_map_json=rule_map_json, simplified_rule_json=simplified_rule_json, simplified_rule_map_json=simplified_rule_map_json, user=user, name=name)


def get_reactive_bar_data(request):
    age = request.form["age"]
    segment = request.form["segment"]
    HGB = request.form["HGB"]
    PLATELET = request.form["PLATELET"]
    Serum_WBC = request.form["Serum_WBC"]
    P_T = request.form["P.T"]
    APTT = request.form["APTT"]
    CCI = request.form["CCI"]
    Elixhauser = request.form["Elixhauser"]
    Revision = request.form["Revision"]
    ASA_2 = request.form["ASA_2"]
    positive_culture = request.form["positive_culture"]
    Serum_CRP = request.form["Serum_CRP"]
    Serum_ESR = request.form["Serum_ESR"]
    Synovial_WBC = request.form["Synovial_WBC"]
    Single_Positive_culture = request.form["Single_Positive_culture"]
    Synovial_PMN = request.form["Synovial_PMN"]
    Positive_Histology = request.form["Positive_Histology"]
    Purulence = request.form["Purulence"]
    arr = [age, segment, HGB, PLATELET, Serum_WBC, P_T, APTT, CCI, Elixhauser,
           Revision, ASA_2, positive_culture, Serum_CRP, Serum_ESR, Synovial_WBC,
           Single_Positive_culture, Synovial_PMN, Positive_Histology, Purulence]
    arr[:] = [float(x) for x in arr]
    return arr


@ app.route('/reactive_diagram', methods=['GET', 'POST'])
def reactive_diagram():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')

    user = User.get_by_uid(uid)
    name = request.args.get('p_id')
    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
                           password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.revision_pji WHERE no_group =" + str(name)
    cur.execute(sql)
    user_data = cur.fetchall()
    conn.close()

    with open("Decision_rule/user_data.json", "w") as file:
        json.dump(user_data, file)
    user_data_json = json.load(open("Decision_rule/user_data.json"))

    global reactive_rule_json
    global result_text
    if request.method == 'GET':
        arr = json.load(open("Decision_rule/reactived_data_onlyvalue.json"))
        predict_data = Vision_compare.tran_df(arr)
        result1 = Vision_compare.stacking_predict(predict_data)
        result_xgb = Vision_compare.xgboost_predict(predict_data)
        result_rf = Vision_compare.rf_predict(predict_data)
        result_nb = Vision_compare.nb_predict(predict_data)
        result_lr = Vision_compare.lr_predict(predict_data)
        print("running start")
        if (result1[0] == 1):
            result_text = "Infected"
        else:
            result_text = "Aseptic"

        reactive_rule_json = json.load(
            open("Decision_rule/reactive_rule.json"))

        reactive_rule_map_json = json.load(
            open("Decision_rule/reactive_rule_map.json"))

        reactive_decision_list_json = json.load(
            open("Decision_rule/decision_rule_reactive_diagram.json"))

        reactive_decision_list_map_json = json.load(
            open("Decision_rule/decision_rule_reactive_diagram_map.json"))

        reactived_data_json = json.load(
            open("Decision_rule/reactived_data.json"))

        Vision_compare.plt_con()
        print("running end")

        return render_template('reactive_diagram.html', result=result_text, reactive_rule_json=reactive_rule_json, reactive_rule_map_json=reactive_rule_map_json,
                               result_xgb=result_xgb[0], result_rf=result_rf[0], result_nb=result_nb[0], result_lr=result_lr[0], user=user, predict_data=arr, reactive_decision_list_json=reactive_decision_list_json, reactive_decision_list_map_json=reactive_decision_list_map_json, user_data_json=user_data_json, reactived_data_json=reactived_data_json, name=name)
    return render_template('reactive_diagram.html', user=user, user_data_json=user_data_json, name=name)


@ app.route('/personal_info')
@limiter.limit(lambda: current_limit)
def personal_info():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')
    # uid = session['user-id']
    if uid != None:
        user = User.get_by_uid(uid)
        if user == None:
            redirect('/auth_login')
        else:
            username = user
    name = request.args.get('p_id')
    personal_result = personal_DecisionPath2.personalDP(int(name))
    if (personal_result == 1):
        result_text = "Infected!"
    else:
        result_text = "Aseptic!"
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    # db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
    # conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
    #                        password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.revision_pji WHERE no_group =" + str(name)
    cur.execute(sql)
    user = cur.fetchall()
    conn.close()
    return render_template('personal_info.html', name=name, u=user, result=result_text, username=username)


@ app.route('/upload_new_data_csv', methods=['GET', 'POST'])
def upload_new_data_csv():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')
    # uid = session['user-id']
    if uid != None:
        user = User.get_by_uid(uid)
        if user == None:
            redirect('/auth_login')
        else:
            username = user
    name = request.args.get('p_id')

    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if not uploaded_file.filename.endswith('.csv') and not uploaded_file.filename.endswith('.xlsx'):
                return "Please upload a CSV or Excel file.", 400
            file_extension = os.path.splitext(uploaded_file.filename)[1]

            if file_extension == '.csv':
                file_path = os.path.join('uploads', 'New_data.csv')
                uploaded_file.save(file_path)
                df = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                file_path = os.path.join('uploads', 'New_data.xlsx')
                uploaded_file.save(file_path)
                df = pd.read_excel(file_path)

            for i in range(1, 26):
                time.sleep(0.3)
                progress = i
                socketio.emit('progress_percent', {'percentage': progress})

            required_columns = ['No.Group', 'PJI/Revision Date', 'Primary, Revision\nnative hip', 'ASA', 'Age',
                                'Serum ESR', 'Serum WBC ', 'Segment (%)', 'HGB', 'PLATELET', 'P.T',
                                'APTT', 'Serum CRP', 'Positive Histology', 'Synovial WBC',
                                '2X positive culture', 'Pulurence', 'Single Positive culture', 'Total CCI',
                                'Total Elixhauser Groups per record']

            if not all(col in df.columns for col in required_columns):
                return jsonify({"status": "error", "message": "Missing required columns"}), 400

            try:
                preprocessing_data(df)
            except Exception as preprocess_error:
                return f"An error occurred during preprocessing: {preprocess_error}", 400
            for i in range(26, 71):
                time.sleep(0.1)
                progress = i
                socketio.emit('progress_percent', {'percentage': progress})
            try:
                upload_to_db()
            except Exception as upload_error:
                return f"An error occurred during database upload: {upload_error}", 400

            for i in range(71, 101):
                time.sleep(0.1)
                progress = i
                socketio.emit('progress_percent', {'percentage': progress})

        socketio.emit('task_completed')
        return "File uploaded and checked successfully."

    return render_template('upload_new_data_csv.html')


client = vonage.Client(key="da423732", secret="PBx69gqMyaAYvTIP")
# sms = vonage.Sms(client)


@ app.route('/message_board', methods=['GET', 'POST'])
def message_board():
    session.permanent = True
    uid = session.get('user-id', None)

    if uid is None:
        print("Session timeout!")
        return redirect('/auth_login')

    user = User.get_by_uid(uid)
    if user is None:
        return redirect('/auth_login')

    if request.method == 'POST':
        username = request.form['username']
        content = request.form['content']

        # Input validation should be here
        # ...

        try:
            conn = pymysql.connect(
                host='127.0.0.1', user='root', password='love29338615', port=3306, db='PJI')
            db_uri_info = urlparse(os.environ.get('DATABASE_URL'))
            # conn = pymysql.connect(host=db_uri_info.hostname, user=db_uri_info.username,
            #                        password=db_uri_info.password, port=db_uri_info.port, db=db_uri_info.path[1:])
            cur = conn.cursor()
            sql_insert = "INSERT INTO message (username, content) VALUES (%s, %s)"
            cur.execute(sql_insert, (username, content))
            conn.commit()
            flash("Leave a message successfully!", "success")

            responseData = client.sms.send_message(
                {
                    "from": "Vonage APIs",
                    "to": "886975286089",
                    "text": f"New message from {username}: {content}",
                    'type': 'unicode',
                }
            )

            if responseData["messages"][0]["status"] == "0":
                print("Message sent successfully.")
                flash("Leave a message and SMS sent successfully!", "success")
            else:
                print(
                    f"Message failed with error: {responseData['messages'][0]['error-text']}")
                flash(
                    f"Leave a message but failed to send SMS: {responseData['messages'][0]['error-text']}", "error")
        except Exception as e:
            flash("Failed to leave a message", "error")
            print("Failed to insert data", e)
        finally:
            conn.close()

    return render_template('message_board.html')

    # conn = pymysql.connect(host='127.0.0.1', user='root',
    #                        password='love29338615', port=3306, db='PJI')
    # cur = conn.cursor()
    # cur.execute("SELECT username, content FROM messages")
    # messages = cur.fetchall()
    # conn.close()
    return render_template('message_board.html')


    # return render_template('message_board.html', messages=messages)
if __name__ == "__main__":
    app.run()
