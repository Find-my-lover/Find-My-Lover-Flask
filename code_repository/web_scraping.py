
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.request
from urllib.request import ProxyHandler, build_opener, install_opener

#웹브라우저 창이 뜨지 않도록 설정
options=webdriver.ChromeOptions()

options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36")
#options.add_argument("user-agent="+user_agent)

men_url="https://www.google.com/search?q=%EB%B0%A9%ED%83%84%EC%86%8C%EB%85%84%EB%8B%A8+%EC%A0%95%EB%A9%B4+%EC%96%BC%EA%B5%B4+%ED%99%94%EB%B3%B4&tbm=isch&ved=2ahUKEwjb2M2L27n7AhVgU_UHHbsYDa8Q2-cCegQIABAA&oq=%EB%B0%A9%ED%83%84%EC%86%8C%EB%85%84%EB%8B%A8+%EC%A0%95%EB%A9%B4+%EC%96%BC%EA%B5%B4+%ED%99%94%EB%B3%B4&gs_lcp=CgNpbWcQA1C01QJYnO4CYMzvAmgBcAB4AIABfYgB_RaSAQQxLjI2mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=WIR4Y9vMLuCm1e8Pu7G0-Ao&bih=714&biw=1536"
proxy = ProxyHandler({})
opener = build_opener(proxy)
opener.addheaders = [('User-Agent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30')]
install_opener(opener)

from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator


@timeout(1000000)
def extract_data_google(url, options, folder_name):
  driver=webdriver.Chrome("chromedriver", options=options)

  driver.get(url)

  SCROLL_PAUSE_TIME=1

  for i in range(1000):
    driver.execute_script("window.scrollBy(0, 5000)")
  imgs=driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
  count=1
  print(imgs)
  for img in imgs:
    img.click()
    time.sleep(2)

    img_url=driver.find_element(By.XPATH,"/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div/div[3]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
    
    urllib.request.urlretrieve(img_url,"/content/drive/MyDrive/Find my lover/"+folder_name+"/"+str(count)+".jpg")
    print(str(count)+"jpg 저장")
    count=count+1



men_url="https://www.google.co.kr/search?source=univ&tbm1=isch&q=%EB%82%A8%EC%9E%90+%EC%97%B0%EC%98%88%EC%9D%B8+%EC%A0%95%EB%A9%B4+%EC%82%AC%EC%A7%84&fir=2k4gFlI6N8ieqM%252Cx2jjmJrmUiHtNM%252C_%253BT2p1scqFP4HO3M%252CiSowa1hAPF_cXM%252C_%253BZgdkRWOrS_srEM%252Cu2L1VQJlocfslM%252C_%253Bpgz3kv31SgTnbM%252CiSowa1hAPF_cXM%252C_%253BAv-QX9b7mdHciM%252CSg6k4ODs_nXxdM%252C_%253B1IBqU9syb7YLVM%252CmZ7WR2Zdb45HCM%252C_%253B0f-ks4Itz545_M%252CWaLndXzTox5gGM%252C_%253BE-0w4ZY9q-Qf1M%252C8urjqEaBFbxDfM%252C_&usg=AI4_-kTF21WpKwaA5Uge8sAta9aK7bOLoQ&sa=X&ved=2ahUKEwj-5bzH37v7AhVKCYgKHdKrAFYQ7Al6BAgIEEg&biw=1536&bih=714&dpr=1.25"
women_url="https://search.naver.com/search.naver?sm=tab_hty.top&where=image&query=%EC%98%88%EC%81%9C+%EC%97%AC%EC%9E%90+%EC%97%B0%EC%98%88%EC%9D%B8+%EC%A0%95%EB%A9%B4+%EC%82%AC%EC%A7%84&oquery=%EC%9E%98%EC%83%9D%EA%B8%B4+%EB%82%A8%EC%9E%90+%EC%97%B0%EC%98%88%EC%9D%B8+%EC%A0%95%EB%A9%B4+%EC%82%AC%EC%A7%84&tqi=hHxvCsp0YiRssOGpRHCssssstAo-269836"

try:
  extract_data_google(men_url, options, "men")
except Exception as e:
  print('timeout occur', str(e))

import cv2
import matplotlib.pyplot as plt

cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'#"haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file) # 정면 얼굴 인식 모델
crop_face=[]
count=0
i=0

while(1):
  if count==64: break
  img = cv2.imread('/content/drive/MyDrive/Find my lover/men/'+str(i+1)+'.jpg') #읽기
  try:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #흑백변환
  except:
    continue
  face_list = cascade.detectMultiScale(img, minSize = (50,50)) #얼굴검출
  i+=1
  print(face_list)
  if len(face_list)==0: 
    print("jump")
    continue
  try:
    print("리스트에 넣어보자")
    test_face_area=face_list[0]
    crop_face.append(img[test_face_area[0]:test_face_area[0]+test_face_area[2], test_face_area[1]:test_face_area[1]+test_face_area[3]])
    count+=1
    

  except:
    pass

  
'''
fig, axs=plt.subplots(8, 8, figsize=(10, 15))
for i in range(8):
  for j in range(8):
      axs[i, j].imshow(crop_face[5*i+j],cmap='gray_r')
      axs[i, j].axis('off')
plt.show()
'''    





