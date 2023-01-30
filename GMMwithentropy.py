import cv2
import numpy as np
import math
import os
import ctypes


def get_entropy(img_,num):
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (int(x/num),int(y/num) ))  # 缩小的目的是加快计算速度
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


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
model = cv2.createBackgroundSubtractorMOG2()
model.setDetectShadows(0)
print(model.setHistory)
framenum =0
history=5
model.setHistory(history)
num=0
flag =0
last=0
while (1):
   # print(num)
    print(history)
    num=num+1
    framenum=framenum+1
    frame=cv2.imread('E:\BOT100\Freeman4\Freeman4\img\\'+str(framenum).zfill(4)+'.jpg')

   # print(frame.shape[0])
#    model.setHistory(history)
    if  flag <= 10:
        history=0.001+(0.2-0.001)*(10-flag)/10
        flag=flag+1
       # model.setHistory(history)
    if flag > 10 and flag<= 110:
        history=(0.001+(0.2-0.001)*math.cos(0.5*math.pi*((flag-10)/100)))
       # model.setHistory(history)
        flag=flag+1
    if(flag>110) :
        history=0.001
    print(history)
    # 第四步：读取视频中的图片，并使用高斯模型进行拟合
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #cv2.imshow('frame', frame)
    #cv2.imwrite('./picture_video/3_origin/' + str(num) + '.jpg', frame)
    # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
    fgmk = model.apply(frame)
    #txt1.write(str(get_entropy(fgmk,1)) + "\n")

    # fgmk = cv2.threshold(fgmk, 244, 255, cv2.THRESH_BINARY)[1]
    #print(get_entropy(frame,10))
    #txt.write(str(get_entropy(fgmk))+"\n")
    now=get_entropy(fgmk,10)
    #now = 0
    if abs(now-last)>0.3:
        flag=0
        history=0.2
       # model.setHistory(history)
        #framenum=0
    last=now
    xx, yy = frame.shape[0:2]
    hh=0
    ww=0
    # fgmk = cv2.GaussianBlur(fgmk, (3, 3), 0)
    # fgmk = cv2.bilateralFilter(src=fgmk, d=0, sigmaColor=50, sigmaSpace=15)
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        # 第七步：进行人的轮廓判断，使用周长，符合条件的画出外接矩阵的方格
        length = cv2.arcLength(c, True)
        (x, y, w, h) = cv2.boundingRect(c)
        if length > 10:
            xx=min(xx,x)
            yy=min(yy,y)
            hh=max(hh,y+h)
            ww=max(ww,x+w)
            # print(x, y, w, h)
    cv2.rectangle(frame, (xx, yy), (ww, hh), (0, 255, 0), 2)
    # 第八步：进行图片的展示
    cv2.imshow('fgmk', fgmk)
    cv2.imshow('mixgauss_gaussblur', frame)


   # cv2.imwrite('./picture_video/1_mask-1/' + str(num) + '.jpg', fgmk)
   # cv2.imwrite('./picture_video/1_after-1/' + str(num) + '.jpg', frame)
    #   cv2.imwrite("D:\\MOGT\\{}.jpg".format(str(xxx)),fgmk)
    #   cv2.imwrite("D:\\MOGTX\\{}.jpg".format(str(yyy)), frame)
    #   xxx=xxx+1
    #   yyy=yyy+1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
