import boto3
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dlib


detector_HOG=dlib.get_frontal_face_detector()

model_path = "C://Users//nanle//ringmybell//Find-My-Lover-Flask//model_save_point//shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(model_path)
#YcrCb space에서 skin색의 상한, 하한
lower = np.array([0,133,77], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)

def crop_head(img):
    try:
        men_dlib_rects=detector_HOG(img, 1)
        try: 
            l=men_dlib_rects[0].left()
            t = men_dlib_rects[0].top()
            r = men_dlib_rects[0].right()
            b = men_dlib_rects[0].bottom()
        except IndexError:
            print("사람이 아닙니당")

        points = landmark_predictor(img, men_dlib_rects[0])
        #print(map(lambda p: (p.x, p.y), points.parts()))
        # face landmark 좌표를 저장
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        #list_points = list(points.parts())
        
        #머리가 있는 y축 좌표
        point_hair_y=int(list_points[30][1]-(list_points[8][1]-list_points[30][1])*4/3)

        img_rgba=cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        instance_img=img_rgba.copy()

        #non_trans.append(instance_img[0][0])

        img_skin=img.copy()
        img_skin=cv2.cvtColor(img_skin, cv2.COLOR_RGB2YCrCb)
        
        skin_mask=cv2.inRange(img_skin, lower, upper)
        skin_index=np.where(skin_mask==255)


        #interpolation을 이용한 턱선 살리기
        drop_list_points=[]
        for i in range(8): #8이 중간!  =>
            x_1, y_1= list_points[i]
            x_2, y_2=list_points[i+1]
            count=0
            for j in range(x_1, x_2):
                drop_list_points.append([j, int(y_1+(y_2-y_1)*count/(x_2-x_1))])
                count+=1 
        for i in range(8, 16): #8이 중간!  =>
            x_1, y_1= list_points[i]
            x_2, y_2=list_points[i+1]
            count=0
            for j in range(x_1, x_2):
                drop_list_points.append([j, int(y_1+(y_2-y_1)*count/(x_2-x_1))])
                count+=1 



        instance_img=img_rgba.copy()
        hair_colors=[]
        #머리색 놓치지 않기 위한 해당 y값에 대한 x평행선의 머리색 다 넣어두기
        for i in range(drop_list_points[0][0], drop_list_points[-1][0], 3):
            hair_colors.append(instance_img[point_hair_y][i])


        #투명화처리할 부분 선택 ------------------------여기부터가 진짜 croping
        
        instance_img=img_rgba.copy()
        background_color=img_rgba[10][10]

        upper_bound=background_color+[50, 50, 50, 0]
        lower_bound=background_color-[50, 50, 50, 0]


        #턱선 밑으로는 다 자르기
        for i in drop_list_points:
            instance_img[i[1]:,i[0]]=(0, 0, 0, 0)
    
    

        #여기가 계산 과부하 오는 부분
        print(instance_img.shape)
        print(b, t)
        for i in range(instance_img.shape[0]):
            for j in range(instance_img.shape[1]):
            

            #양 옆에 자르기
                if j>drop_list_points[-1][0] or j<drop_list_points[0][0]:
                    flag=False
                    #머리 윗부분에 대해서 머리살리기
                    if i<drop_list_points[-1][-1]:
                        for hair_color in hair_colors:
                            if all(instance_img[i][j] <= hair_color+[30, 30, 30, 0]) and all(instance_img[i][j]>=hair_color-[30, 30, 30, 0]):
                                flag=True
                        if flag==True: continue
                    instance_img[i][j]=(0, 0, 0, 0)
                #턱선 안쪽에서 얼굴 부분
                elif i>t-(b-t)*1/3 and i<b:
                    continue

                if i>t-(b-t)*1/3 and i<b+(b-t)*1/4 and j>l and j<r and i in skin_index[0] and j in skin_index[1]:
                    continue
                
                if all(instance_img[i][j] <= upper_bound) and all(instance_img[i][j]>=lower_bound):

                    instance_img[i][j]=(0, 0, 0, 0)
    except IndexError:
        print("index error jump")
    return instance_img
  
def s3_connection():
    try:
        session=boto3.Session(
            region_name="ap-northeast-2",
            aws_access_key_id="AKIASR6XYJ5YJGKZVUHZ",
            aws_secret_access_key="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"
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return session


def read_image_from_s3(filename):
    bucket=s3.Bucket("my.image.bucket")
    object=bucket.Object(filename)
    response=object.get()
    file_stream=response["Body"]
    img=Image.open(file_stream)
    img=cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    #cv2.imshow("img", img)
    #cv2.waitKey(500)
    return img;
s3=s3_connection().resource("s3")
cv2.imshow("result", cv2.cvtColor(crop_head(read_image_from_s3("partner/predicted_men/cat/43.jpg")), cv2.COLOR_BGR2RGB))
cv2.waitKey(50000)
#plt.imshow(read_image_from_s3("partner/predicted_men/cat/117.jpg"))

#partner_img=s3.get_object(Bucket="my.image.bucket", Key="partner/predicted_men/cat/117.jpg")
#print(partner_img)
#img_show=cv2.cvtColor(cv2.imread(partner_img), cv2.COLOR_BGR2RGB)
#cv2.imshow(img_show)
