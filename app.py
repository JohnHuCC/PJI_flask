from sklearn.metrics import confusion_matrix
import Vision_compare
from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import pymysql

db = SQLAlchemy()
app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:love29338615@127.0.0.1:3306/PJI"

db.init_app(app)

import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df_data = pd.read_csv(
    'PJI_Dataset/PJI_train.csv')

PJI_X = pd.DataFrame([df_data["Age"],
                      df_data["Segment"],
                      df_data["HGB"],
                      df_data["PLATELET"],
                      df_data["Serum "],
                      df_data["P.T"],
                      df_data["APTT"],
                      df_data["Total CCI"],
                      df_data["Total Elixhauser Groups per record"],
                      df_data["Primary,Revision,native hip"],
                      df_data["ASA_2"],
                      df_data["2X positive culture"],
                      df_data["Serum CRP"],
                      df_data["Serum ESR"],
                      df_data["Synovial WBC"],
                      df_data["Single Positive culture"],
                      df_data["Synovial_PMN"],
                      df_data["Positive Histology"],
                      df_data["Purulence"],
                      ]).T
PJI_y = df_data["Group"]

train_X, test_X, train_y, test_y = train_test_split(
    PJI_X, PJI_y, test_size=0.3)

# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)
forest_fit = forest.fit(train_X, train_y)
joblib.dump(forest, 'RF_model')

# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def tran_df(arr):
    predict_data = pd.DataFrame(
        np.array([arr]), columns=['Age', 'Segment', 'HGB', 'PLATELET', 'Serum ', 'P.T', 'APTT', 'Total CCI', 'Total Elixhauser Groups per record', 'Primary,Revision,native hip', 'ASA_2',
                                  '2X positive culture', 'Serum CRP', 'Serum ESR', 'Synovial WBC', 'Single Positive culture', 'Synovial_PMN', 'Positive Histology', 'Purulence'
                                  ])
    print(predict_data)
    return predict_data


# predict_data = tran_df(arr)


def rf_predict(df):
    loaded_model = joblib.load('RF_model')
    result = loaded_model.predict(df)
    print(result)
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    name = 1
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.pji_small"
    cur.execute(sql)
    u = cur.fetchall()
    conn.close()

    if request.method == 'POST':
        dict_p = request.get_json()
        print(dict_p["patientID"])
        if dict_p["btnID"] == "0":
            return url_for('personal_info', p_id=dict_p["patientID"])
        elif dict_p["btnID"] == "1":
            return url_for('model_diagnosis', p_id=dict_p["patientID"])
        elif dict_p["btnID"] == "2":
            return url_for('reactive_diagram', p_id=dict_p["patientID"])

    return render_template('index.html', u=u, name="")

    # sql_cmd = """
    #     select *
    #     from PJI.pji_small
    #     """
    # query_data = db.engine.execute(sql_cmd)
    # return 'ok'


if __name__ == "__main__":
    app.run()


@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')


@app.route("/chart2")
def chart2():
    return render_template('chart2.html')


@app.route("/chart3")
def chart3():
    return render_template('chart3.html')


@app.route('/model_diagnosis')
def model_diagnosis():
    decision_list_file = open("decision_list_file.txt", "r")
    decision_list_file_content = decision_list_file.read().strip()
    decision_list = decision_list_file_content.split("\n")
    decision_list_file.close()
    return render_template('model_diagnosis.html', decision_list=decision_list)


@app.route('/reactive_diagram', methods=['GET', 'POST'])
def reactive_diagram():
    if request.method == 'POST':
        age = request.form["age"]
        segment = request.form["segment"]
        HGB = request.form["HGB"]
        PLATELET = request.form["PLATELET"]
        Serum = request.form["Serum"]
        P_T = request.form["P.T"]
        APTT = request.form["APTT"]
        CCI = request.form["CCI"]
        Elixhauser = request.form["Elixhauser"]
        Rivision = request.form["Rivision"]
        ASA_2 = request.form["ASA_2"]
        positive_culture = request.form["positive_culture"]
        Serum_CRP = request.form["Serum_CRP"]
        Serum_ESR = request.form["Serum_ESR"]
        Synovial_WBC = request.form["Synovial_WBC"]
        Single_Positive_culture = request.form["Single_Positive_culture"]
        Synovial_PMN = request.form["Synovial_PMN"]
        Positive_Histology = request.form["Positive_Histology"]
        Purulence = request.form["Purulence"]
        print("age:", age)
        arr = [age, segment, HGB, PLATELET, Serum, P_T, APTT, CCI, Elixhauser,
               Rivision, ASA_2, positive_culture, Serum_CRP, Serum_ESR, Synovial_WBC,
               Single_Positive_culture, Synovial_PMN, Positive_Histology, Purulence]
        predict_data = Vision_compare.tran_df(arr)
        result1 = Vision_compare.rf_predict(predict_data)
        print("running start")
        Vision_compare.plt_con()
        print("running end")
        return render_template('reactive_diagram.html', result=result1)

    return render_template('reactive_diagram.html')


@app.route('/personal_info')
def personal_info():
    name = request.args.get('p_id')
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.pji_small WHERE ID =" + str(name)
    cur.execute(sql)
    u = cur.fetchall()
    conn.close()
    return render_template('personal_info.html', name=request.args.get('p_id'), u=u)


@app.route('/user/<name>')
def thisDemo(name):
    return "my name is %s" % name


@app.route('/<user>')
def welcome(user):
    # username 為 KEY 值，html 檔會用到；user 為 value，是要帶入的值
    return render_template('personal_info.html', username=user)


@app.route('/testapi', methods=['POST'])
def testapi():
    dict_p = request.get_json()
    print(dict_p["patientID"])
    return "Okk"
