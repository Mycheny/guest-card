import cv2
from pyzbar.pyzbar import decode

a = decode(cv2.imread('timg1.jpg'))
print(a)
