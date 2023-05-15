from flask import Flask
from flask_restx import Resource, Api
import sys

sys.path.append("C:/Users/nanle/ringmybell/Find-My-Lover-Flask/pybo/service/")
from photo import *

app = Flask(__name__)
api = Api(app)


@api.route('/test')  # url pattern으로 name 설정
class test(Resource):
    def post(self):  # 멤버 함수의 파라미터로 name 설정
        upload_file()
    

if __name__=='__main__':
        app.run(debug=False, host="127.0.0.1", port=5000)