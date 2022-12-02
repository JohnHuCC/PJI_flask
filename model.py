from db import db
import bcrypt
# from itsdangerous import TimedJSONWebSignatureSerializer
from flask import current_app


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    session_hash = db.Column(db.String(30), nullable=True)

    def get_by_name(name):
        return User.query.filter_by(name=name).first()

    def get_by_uid(session_hash):
        return User.query.filter_by(session_hash=session_hash).first()

    def check_password(self, passwd):
        """
        密碼驗證，驗證使用者輸入的密碼跟資料庫內的加密密碼是否相符
        :param password: 使用者輸入的密碼
        :return: True/False
        """
        return bcrypt.check_password_hash(self.password, passwd)

    def load_user(user_id):
        return User.query.get(int(user_id))

    # def create_reset_token(self, expires_in=3600):
    #     """
    #     提供申請遺失密碼認證使用的token
    #     :param expires_in: 有效時間(秒)
    #     :return:token
    #     """
    #     s = TimedJSONWebSignatureSerializer(
    #         current_app.config['SECRET_KEY'], expires_in=expires_in)
    #     return s.dumps({'reset_id': self.id})
