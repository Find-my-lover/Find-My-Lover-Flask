import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from PIL import Image
from tqdm import tqdm

import os

men_folder="/content/drive/MyDrive/Find my lover/predicted_men/"
labels=os.listdir(men_folder)
img_list=dict.fromkeys(labels)
#부엉이상, 사슴상, 고양이상, 늑대상, 사자상, 염소상, 말상, 고양이상, 쥐상, 캥거루상, 당나귀상, 코알라상, 여우상
for i in labels:
  animal_type=men_folder+i
  img_list[i]=[]
  for (root, _, files) in os.walk(animal_type):
    for file in files:
      img_list[i].append(cv2.cvtColor(cv2.imread(root+"/"+file), cv2.COLOR_BGR2RGB))


detector_HOG=dlib.get_frontal_face_detector()

model_path = "/content/drive/MyDrive/Find my lover/shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(model_path)


'''
img_list=[]
for i in range(50):
  image_path=str(i+1)+".png"
  img_list.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
'''
fig, axs=plt.subplots(10, 5, figsize=(20, 20))
for i in range(10):
  for j in range(5):
    axs[i, j].imshow(img_list[i*5+j])

plt.show()

safe_img_list=[]
cropping_face=[]


#YcrCb space에서 skin색의 상한, 하한
lower = np.array([0,133,77], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)

fig, axs=plt.subplots(len(safe_img_list), 2, figsize=(20, 200))
for i in range(len(safe_img_list)):
  axs[i, 0].imshow(safe_img_list[i])
  axs[i, 1].imshow(cropping_face[i])



safe_img_type=dict.fromkeys(labels)
cropping_img_type=dict.fromkeys(labels)
label="wolf"
safe_img_type[label]=[]
cropping_img_type[label]=[]
for img in tqdm(img_list[label]):
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
    cropping_img_type["wolf"].append(instance_img)
      
  except IndexError:
    print("index error jump")
  
    
fig, axs=plt.subplots(len(cropping_img_type["wolf"]), 1, figsize=(20, 400))
for i in range(len(cropping_img_type["wolf"])):
  axs[i].imshow(cv2.cvtColor(cropping_img_type["wolf"][i], cv2.COLOR_BGR2RGBA))
  axs[i].subtitle()
plt.title(label)

plt.show()
