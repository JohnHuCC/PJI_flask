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
        confusion_matrix = Vision_compare.plt_con()
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