#  追加import
from wtforms import BooleanField, EmailField, PasswordField, SubmitField, validators
from flask_wtf import FlaskForm
from model import User
from wtforms.validators import ValidationError


class FormChangePWD(FlaskForm):
    """
    使用者變更密碼
    舊密碼、新密碼與新密碼確認
    """
    #  舊密碼
    password_old = PasswordField('PassWord', validators=[
        validators.DataRequired()
    ])
    #  新密碼
    password_new = PasswordField('PassWord', validators=[
        validators.DataRequired(),
        validators.Length(5, 10),
        validators.EqualTo('password_new_confirm',
                           message='PASSWORD NEED MATCH')
    ])
    #  新密碼確認
    password_new_confirm = PasswordField('Confirm PassWord', validators=[
        validators.DataRequired()
    ])
    submit = SubmitField('Change Password')


class FormResetPasswordMail(FlaskForm):
    """應用於密碼遺失申請時輸入郵件使用"""
    email = EmailField('Email', validators=[
        validators.DataRequired(),
        validators.Length(5, 30),
        validators.Email()
    ])
    submit = SubmitField('Send Confirm EMAIL')

    def validate_email(self, field):
        """
        驗證是否有相關的EMAIL在資料庫內，若沒有就不寄信
        """
        if not UserRegister.query.filter_by(email=field.data).first():
            raise ValidationError('No Such EMAIL, Please Check!')
