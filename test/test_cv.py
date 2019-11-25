import base64
import urllib.parse, urllib.request, http.client

import cv2
import requests
import json

import time

url = 'http://172.16.205.48:8093/xiaoi/identify'
# url = 'http://127.0.0.1:5000/xiaoi/identify'

cap = cv2.VideoCapture(0)
while True:
    start = time.time()
    s, fream = cap.read()
    fream = cv2.flip(fream, 1)
    cv2.imshow('fream', fream)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
