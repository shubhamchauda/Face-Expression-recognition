import urllib3
import cv2
import numpy as np
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.29.152:8080/video'

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    print(frame)
    if frame is not None:
        cv2.imshow('frame',frame)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()