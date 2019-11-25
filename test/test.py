import base64

import cv2
import requests
import json

# url = 'http://192.168.160.206:8092/distance'
import time

url = 'http://127.0.0.1:8092/distance'
# url = 'http://127.0.0.1:10000/websocket'
image1=cv2.imread('11.png')
image1 = cv2.resize(image1,(260,400))
# image1=open('IMG_20180503_172643.jpg', 'rb').read()
ret, image1 = cv2.imencode('.jpg', image1)
image1=image1.tobytes()
# image = base64.b64encode(image.read()).decode()
image2 = base64.b64encode(image1).decode()


data = json.dumps({"body": "yes", "age_gender": "yes", "close": "no"})
while True:
    r = requests.post(url, data=data)
    a, b, c = r.text, r.content, r.json()
    # print(r.text,r.content,r.json())
    print(r.json())
    # time.sleep(2)
