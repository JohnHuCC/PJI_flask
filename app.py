from form import FormChangePWD, FormResetPasswordMail
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import joblib
import pandas as pd
import numpy as np
import Vision_compare
from flask import Flask, render_template, request, url_for, redirect, flash, session
import json
from flask_login import logout_user,  current_user, login_required
from db import db
from flask_bootstrap import Bootstrap5
from flask_migrate import Migrate
from model import User
import os
from datetime import timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=31)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:love29338615@127.0.0.1:3306/PJI"

app.config['WTF_CSRF_ENABLED'] = False

bootstrap = Bootstrap5(app)
db.init_app(app)
migrate = Migrate(app, db)

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
    session.permanent = True
    uid = session.get['user-id']
    if uid != None:
        User.get_by_uid(uid)
        pass

    name = 1
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.revision_pji"
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


@app.route("/auth_login", methods=['GET', 'POST'])
def auth_login():
    if request.method == 'GET':
        return render_template('auth_login.html')
    elif request.method == 'POST':
        username = request.form['username']
        user = User.get_by_name(username)
        if user.check_password(request.form['password']):
            session_hash = username + os.urandom(30)
            user.session_hash = session_hash
            db.session.commit()
            session['user-id'] = session_hash
            return redirect('/')
        else:
            return redirect('/auth_login')


def next_is_valid(url):
    """
    為了避免被重新定向的url攻擊，必需先確認該名使用者是否有相關的權限。
    舉例來說，如果使用者調用了一個刪除所有資料的uri，那就GG了是吧。
    :param url: 重新定向的網址
    :return: boolean
    """
    return True


@app.route('/logout')
def logout():
    logout_user()
    flash('Log Out See You.')
    return redirect(url_for('login'))


# @app.before_request
# def before_request():
#     """
#     在使用者登入之後，需做一個帳號啟動得驗證，始得以向下展開相關的應用。
#     條件一：需登入
#     條件二：未啟動
#     條件三：endpoint不等於static，這是避免靜態資源的取用異常，如icon、js、css等..
#     條件四：必需加入相關例外清單
#     :return:
#     """
#     app.secret_key = 'xxxxyyyyyzzzzz'
#
#     login_manager = LoginManager()
#     login_manager.init_app(app)
#     login_manager.login_view = 'login'
#     # app.run(debug=DEBUG, host=HOST, port=PORT)
#     if (current_user.is_authenticated and
#         not current_user.confirm and
#         request.endpoint not in ['re_userconfirm', 'logout', 'user_confirm'] and
#             request.endpoint != 'static'):
#         #  條件滿足就引導至未啟動說明
#         flash('Hi, please activate your account first. Your endpoint:%s' %
#               request.endpoint)
#         return render_template('unactivate.html')


# @app.after_request
# def after_request(response):
#     print('after request finished')
#     print(request.url)
#     response.headers['key'] = 'value'
#     return response


@app.route('/reusreconfirm')
@login_required
def re_userconfirm():
    """
    當使用者點擊重新寄送的時候就引導到這個route
    因為已經使用current_user綁定user了，所以可以直接透過current_user使用user的相關方法
    重新寄送啟動信件必需要登入狀態
    :return:
    """
    #  產生用戶認證令牌
    token = current_user.create_confirm_token()
    #  寄出帳號啟動信件
    send_mail(sender='Your Mail@hotmail.com',
              recipients=['Your Mail@gmail.com'],
              subject='Activate your account',
              template='author/mail/welcome',
              mailtype='html',
              user=current_user,
              token=token)
    flash('Please Check Your Email..')
    return redirect(url_for('index'))


@app.route('/changepassword', methods=['GET', 'POST'])
@login_required
def changepassword():
    form = FormChangePWD()
    if form.validate_on_submit():
        #  透過current_user來使用密碼認證，確認是否與現在的密碼相同
        if current_user.check_password(form.password_old.data):
            current_user.password = form.password_new.data
            db.session.add(current_user)
            db.session.commit()
            flash('You Have Already Change Your Password, Please Login Again.')
            return redirect(url_for('logout'))
        else:
            flash('Wrong Password...')
    return render_template('changepassword.html', form=form)


@app.route('/resetpassword', methods=['GET', 'POST'])
def reset_password():
    #  只允許未登入的匿名帳號可以申請遺失密碼
    if not current_user.is_anonymous:
        return redirect(url_for('index'))

    form = FormResetPasswordMail()
    if form.validate_on_submit():
        #  取得使用者資料
        user = UserRegister.query.filter_by(email=form.email.data).first()
        if user:
            #  產生一個token
            token = user.create_reset_token()
            #  寄出通知信
            # send_mail(sender='Yourmail@hotmail.com',  # 嫌麻煩就直接透過參數來做預設
            #           recipients=[user.email],
            #           subject='Reset Your Password',
            #           template='author/mail/resetmail',
            #           mailtype='html',
            #           user=current_user,
            #           token=token)
            flash('Please Check Your Email. Then Click link to Reset Password')
            #  寄出之後將使用者導回login，並且送出flash message
            return render_template(url_for('login'))
    return render_template('resetpasswordemail.html', form=form)


@app.route('/resetpassword/<token>', methods=['GET', 'POST'])
def reset_password_recive(token):
    """使用者透過申請連結進來之後，輸入新的密碼設置，接著要驗證token是否過期以及是否確實有其user存在
    這邊使用者並沒有登入，所以記得不要很順手的使用current_user了。
    """
    if not current_user.is_anonymous:
        return redirect(url_for('index'))

    form = FormResetPasswordMail()

    if form.validate_on_submit():
        user = UserRegister()
        data = user.validate_confirm_token(token)
        if data:
            #  如果未來有需求的話，還要確認使用者是否被停權了。
            #  如果是被停權的使用者，應該要先申請復權。
            #  下面注意，複製過來的話記得改一下id的取得是reset_id，不是user_id
            user = UserRegister.query.filter_by(
                id=data.get('reset_id')).first()
            #  再驗證一次是否確實的取得使用者資料
            if user:
                user.password = form.password.data
                db.session.commit()
                flash('Sucess Reset Your Password, Please Login')
                return redirect(url_for('login'))
            else:
                flash('No such user, i am so sorry')
                return redirect(url_for('login'))
        else:
            flash('Wrong token, maybe it is over 24 hour, please apply again')
            return redirect(url_for('login'))
    return render_template('author/resetpassword.html', form=form)


@app.route("/auth_password")
def auth_password():
    return render_template('auth_password.html')


@app.route("/auth_register")
def auth_register():
    return render_template('auth_register.html')


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
    decision_list_json = json.load(open("decision_rule.json"))
    rule_map_json = json.load(open("decision_rule_map.json"))
    return render_template('model_diagnosis.html', decision_list=decision_list, decision_list_json=decision_list_json, rule_map_json=rule_map_json)
    # return render_template('model_diagnosis.html', decision_list_json=decision_list_json, rule_map_json=rule_map_json)


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
        arr = [age, segment, HGB, PLATELET, Serum, P_T, APTT, CCI, Elixhauser,
               Rivision, ASA_2, positive_culture, Serum_CRP, Serum_ESR, Synovial_WBC,
               Single_Positive_culture, Synovial_PMN, Positive_Histology, Purulence]
        arr[:] = [int(x) for x in arr]
        predict_data = Vision_compare.tran_df(arr)
        result1 = Vision_compare.stacking_predict(predict_data)
        if (result1[0] == 1):
            result_text = "You got infected!"
        else:
            result_text = "You are safe!"
        print("running start")
        Vision_compare.plt_con()
        print("running end")
        return render_template('reactive_diagram.html', result=result_text)

    return render_template('reactive_diagram.html')


@app.route('/personal_info')
def personal_info():
    name = request.args.get('p_id')
    print(name)
    conn = pymysql.connect(host='127.0.0.1', user='root',
                           password='love29338615', port=3306, db='PJI')
    cur = conn.cursor()
    sql = "SELECT * FROM PJI.revision_pji WHERE ID =" + str(name)
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
