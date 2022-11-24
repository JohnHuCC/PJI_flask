#  追加import
from wtforms import BooleanField, EmailField, PasswordField, SubmitField, validators
from flask_wtf import FlaskForm


class FormLogin(FlaskForm):
    """
    使用者登入使用
    以email為主要登入帳號，密碼需做解碼驗證
    記住我的部份透過flask-login來實現
    """

    email = EmailField('Email', validators=[
        validators.DataRequired(),
        validators.Length(5, 30),
        validators.Email()
    ])

    password = PasswordField('PassWord', validators=[
        validators.DataRequired()
    ])

    remember_me = BooleanField('Keep Logged in')

    submit = SubmitField('Log in')
