from flask import Flask, request
from flask_restx import Resource, Api
import sys
import json

sys.path.append("C:/Users/nanle/ringmybell/Find-My-Lover-Flask/pybo/service/")
#from photo import *

app = Flask(__name__)
api = Api(app)

@api.route("/test_result", methods=["POST"])
class test_result(Resource):
    def post(self):
        data=request.get_json()
        print(data)
        #data=data.decode("UTF-8")
       
        print(data["message"])
        return  "ok"
        
        #yield "string-boot"

# @api.route('/test')  # url pattern으로 name 설정
# class test(Resource):
#     def post(self, email):  # 멤버 함수의 파라미터로 name 설정
#         request.json.get("")
#         upload_file(email)
    

if __name__=='__main__':
        app.run(debug=False, host="127.0.0.1", port=5000)