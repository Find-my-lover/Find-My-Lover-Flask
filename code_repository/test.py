from flask import Flask, request
from flask_restx import Resource, Api

app = Flask(__name__)
api = Api(app)

import urllib.request

app=Flask(__name__)
api=Api(app)

#
@api.route("/test_result")
def test_result():
    
    return "ok" 
    
@app.route("/test")
class test(Resource):
    def post(self):
        user_info=request.json.get("data")
    
        return user_info


if __name__=='__main__':
        app.run(debug=False, host="127.0.0.1", port=5000)










# import boto3
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import dlib

# def s3_connection():
#     try:
#         session=boto3.Session(
#             region_name="ap-northeast-2",
#             aws_access_key_id="AKIASR6XYJ5YJGKZVUHZ",
#             aws_secret_access_key="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"
#         )
#     except Exception as e:
#         print(e)
#     else:
#         print("s3 bucket connected!")
#         return session

# def read_image_from_s3(filename):
#     bucket=s3.Bucket("my.image.bucket")
#     object=bucket.Object(filename)
#     response=object.get()
#     file_stream=response["Body"]
#     img=Image.open(file_stream)
#     img=cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
#     #cv2.imshow("img", img)
#     #cv2.waitKey(500)
#     return img

# s3=s3_connection().resource("s3")


# img=read_image_from_s3("partner/predicted_men/cat/43.jpg")
# print(type(img))

# #cv2.imshow("test", img[0:100])
# #cv2.waitKey(5000)
# img=cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
# cv2.imshow("test_1", img[0:100, 0:500])
# cv2.waitKey(5000)
