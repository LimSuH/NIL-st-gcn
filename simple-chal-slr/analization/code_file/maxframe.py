import os
import numpy as np
import cv2

video = '/dataset/KETI_SignLanguage/Video/0001~3000/KETI_SL_0000000001.avi'
cap = cv2.VideoCapture(video)
npy = np.load('/dataset/KETI_SignLanguage/Keypoints-removal/KETI_SL_0000000005.avi.npy')
i = 0
while cap.isOpened():
    success, image = cap.read()
    img = cv2.circle(image, (0,0), 63, (0,0,255), -1)
    x = int(npy[i][0][0])
    y = int(npy[i][0][1])
    print(y)
    img = cv2.circle(img, (x,y), 30, (0,255,0), -1)
    print(x, y)
    cv2.imwrite('test.png', img)
    i +=1
    
