import cv2
import numpy as np
import math
import os
import ctypes
import utils
import torch

def get_entropy(img_,num):
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (int(x/num),int(y/num) ))
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[int(val)] = float(tmp[int(val)] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

rnet, agent =utils.get_model('R32_C10')
checkpoint = torch.load('*')
agent.load_state_dict(checkpoint['agent'])
rnet.load_state_dict(checkpoint['resnet'])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
model = cv2.createBackgroundSubtractorMOG2()
model.setDetectShadows(0)
framenum =0
history=1000
model.setHistory(history)
num=0
flag =1
last=0
while (1):

    print(history)
    num=num+1
    framenum=framenum+1
    frame = cv2.imread('**')
    if  flag <= 8:
        history=0.001+(0.2-0.001)*(10-flag)/10
        flag=flag+1
    if flag > 8 and flag<= 18:
        history=(0.001+(0.2-0.001)*math.cos(0.5*math.pi*((flag-8)/10)))
        flag=flag+1
    if(flag>18) :
        history=0.001
    model.setHistory(int(1/history))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmk = model.apply(frame)
    now=get_entropy(frame,5)
    if abs(now-last)>0.05:
        flag=1
        history=0.001
    last=now
    xx, yy = frame.shape[0:2]
    hh=0
    ww=0
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #one object
    for c in contours:
        length = cv2.arcLength(c, True)
        (x, y, w, h) = cv2.boundingRect(c)
        if length > 30:
            xx=min(xx,x)
            yy=min(yy,y)
            hh=max(hh,y+h)
            ww=max(ww,x+w)

    cropImg = frame[yy:hh,xx:ww ]
    cropImg = cv2.resize(cropImg, (32,64))
    probs, _ = agent(cropImg)
    policy = probs.clone()
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0
    preds = np.argmax(rnet.forward_single(cropImg, policy.data.squeeze(0)))
    if preds==1:
        cv2.rectangle(frame, (xx, yy), (ww, hh), (0, 255, 0), 2)
    #muti objects
    # for c in contours:
    #     length = cv2.arcLength(c, True)
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     if length > 30:
    #         cropImg = frame[y:(y+h),x:(x+w)]
    #         cropImg = cv2.resize(cropImg, (32,64))
    #
    #         probs, _ = agent(cropImg)
    #         policy = probs.clone()
    #         policy[policy < 0.5] = 0.0
    #         policy[policy >= 0.5] = 1.0
    #         preds = np.argmax(rnet.forward_single(cropImg, policy.data.squeeze(0)))
    #         if preds==1:
    #             cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)
    cv2.imshow('fgmk', fgmk)
    cv2.imshow('mixgauss_gaussblur', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
