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
import smote

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=31)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:love29338615@127.0.0.1:3306/PJI"

app.config['WTF_CSRF_ENABLED'] = False
app.permanent_session_lifetime = timedelta(minutes=10)
bootstrap = Bootstrap5(app)
db.init_app(app)
migrate = Migrate(app, db)


def tran_df(arr):
    predict_data = pd.DataFrame(
        np.array([arr]), columns=['Age', 'Segment', 'HGB', 'PLATELET', 'Serum ', 'P.T', 'APTT', 'Total CCI', 'Total Elixhauser Groups per record', 'Primary,Revision,native hip', 'ASA_2',
                                  '2X positive culture', 'Serum CRP', 'Serum ESR', 'Synovial WBC', 'Single Positive culture', 'Synovial_PMN', 'Positive Histology', 'Purulence'
                                  ])
    return predict_data


@app.route('/', methods=['GET', 'POST'])
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

    return render_template('index.html', u=u, name=user)


@app.route('/train_new_data', methods=['GET', 'POST'])
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

    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
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


if __name__ == "__main__":
    app.run()


@app.route("/auth_login", methods=['GET', 'POST'])
def auth_login():
    if request.method == 'GET':
        return render_template('auth_login.html')
    elif request.method == 'POST':
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


@app.route("/auth_register", methods=['GET', 'POST'])
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


@app.route('/logout')
def logout():
    session['user-id'] = False
    # flash('Log Out See You.')
    return redirect('/auth_login')


@app.route("/auth_password")
def auth_password():
    return render_template('auth_password.html')


@app.route("/pick_new_data")
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
    personal_result = smote.personalDP(int(name))
    if (personal_result == 1):
        result_text = "Infected"
    else:
        result_text = "Aseptic"
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
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
        open("decision_rule_"+name+".json"))
    rule_map_json = json.load(
        open("decision_rule_map_"+name+".json"))
    return render_template('pick_new_data.html', name=name, u=new_data, result=result_text, username=username, decision_list=decision_list, decision_list_json=decision_list_json, rule_map_json=rule_map_json)


@app.route("/upload_new_data", methods=['GET', 'POST'])
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
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    cur = conn.cursor()
    sql_delete = f"DELETE FROM PJI.pji_new_data WHERE no_group = {str(name)};"
    sql_insert = f"INSERT INTO PJI.revision_pji SELECT * FROM PJI.pji_new_data WHERE no_group = {str(name)};"
    cur.execute(sql_insert)
    cur.execute(sql_delete)
    conn.commit()
    conn.close()
    return url_for('train_new_data')


@app.route("/chart3")
def chart3():
    return render_template('chart3.html')


@app.route('/model_diagnosis')
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
    decision_list_file = open("decision_list_file.txt", "r")
    decision_list_file_content = decision_list_file.read().strip()
    decision_list = decision_list_file_content.split("\n")
    decision_list_file.close()
    decision_list_json = json.load(
        open("Decision_rule/decision_rule_"+name+".json"))
    rule_map_json = json.load(
        open("Decision_rule/decision_rule_map_"+name+".json"))
    return render_template('model_diagnosis.html', decision_list=decision_list, decision_list_json=decision_list_json, rule_map_json=rule_map_json, user=user)


@app.route('/reactive_diagram', methods=['GET', 'POST'])
def reactive_diagram():
    session.permanent = True
    uid = None
    try:
        uid = session['user-id']
    except KeyError:
        print("Session timeout!")
        return redirect('/auth_login')
    user = User.get_by_uid(uid)
    global reactive_rule_json
    global result_text
    if request.method == 'POST':
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
        predict_data = Vision_compare.tran_df(arr)
        result1 = Vision_compare.stacking_predict(predict_data)
        result_xgb = Vision_compare.xgboost_predict(predict_data)
        result_rf = Vision_compare.rf_predict(predict_data)
        result_nb = Vision_compare.nb_predict(predict_data)
        result_lr = Vision_compare.lr_predict(predict_data)

        if (result1[0] == 1):
            result_text = "Infected"
        else:
            result_text = "Aseptic"
        reactive_rule_json = json.load(
            open("Decision_rule/reactive_rule.json"))

        reactive_rule_map_json = json.load(
            open("Decision_rule/reactive_rule_map.json"))
        print("running start")
        Vision_compare.plt_con()
        print("running end")
        return render_template('reactive_diagram.html', result=result_text, reactive_rule_json=reactive_rule_json, reactive_rule_map_json=reactive_rule_map_json,
                               result_xgb=result_xgb[0], result_rf=result_rf[0], result_nb=result_nb[0], result_lr=result_lr[0], user=user, predict_data=arr)
        # return render_template('reactive_diagram.html', result=result_text, reactive_rule_json=reactive_rule_json, reactive_rule_map_json=reactive_rule_map_json,
        #                        result_xgb=result_xgb[0], result_rf=result_rf[0], result_nb=result_nb[0], result_lr=result_lr[0], user=user)
    return render_template('reactive_diagram.html', user=user)


@ app.route('/personal_info')
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
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.revision_pji WHERE no_group =" + str(name)
    cur.execute(sql)
    user = cur.fetchall()
    conn.close()
    return render_template('personal_info.html', name=name, u=user, result=result_text, username=username)


@app.route('/progress')
def progress():
    tempfile = open("progress.tmp", "r").read()
    tempfile.close()
    return tempfile
