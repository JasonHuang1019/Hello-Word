# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:11:20 2021

@author: Jason

剛剛我看到你用2種方法找核質比，以這樣"優質"的細胞影像，我覺得下面的方式會更直接，且不會有錯。

1. 定義整張影像，低於灰階20%為細胞核範圍。
2. 取soble，並去除細胞核區域，找出低灰階pixle(定義為細胞質的邊)
3. 定義細胞核的"重心"為中點(A)
4. 計算(A)與每一個細胞質邊的像素(2.)，計算其向量長度，並將每一點的向量長度放在一陣列中。
5. 計算這陣列裡的"中位數"，即為細胞質的半徑(R)
6. 核質比 = (細胞核面積)/(pi*R^2)


cv2.threshold (src, thresh, maxval, type)
cv2.threshold (源图片, 阈值, 填充色, 阈值类型)
src：源图片，必须是单通道
thresh：阈值，取值范围0～255
maxval：填充色，取值范围0～255
type：阈值类型，具体见下表

阈值	 小于阈值的像素点	大于阈值的像素点
0	 置0             	置填充色
1	 置填充色	        置0
2	 保持原色	        置灰色
3	 置0	            保持原色
4	 保持原色	        置0
"""

    
#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math


def otsu(path):
    # os.chdir(r'C:\Users\Jason\Desktop\otsu')
    # plt.close('all')
    #0~255 暗到亮
    img = cv2.imread(path)
    # image = cv2.imread(r"C:\Users\Jason\Desktop\otsu\C20-73-4.png")
    
    # image = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(gray,'gray')

#1. 定義整張影像，低於灰階20%為細胞核範圍。(改為高於最黑的點 20%為細胞核範圍)
    gray_threshold = gray.min()*1.2
    ret1,thresh_1 = cv2.threshold(gray.copy(),gray_threshold,255,2)
    # plt.figure()
    plt.subplot(3,1,2)
    plt.imshow(thresh_1,'gray')
    plt.title("threshold : " +f"{gray_threshold}")

#2. 取sobel，並去除細胞核區域，找出低灰階pixle(定義為細胞質的邊)

# ret, th1 = cv2.threshold(thresh_1.copy(), 127, 255, cv2.THRESH_BINARY)
# # plt.figure()
# # plt.imshow(th1,'gray')

# x = cv2.Sobel(th1,cv2.CV_16S,1,0)
# y = cv2.Sobel(th1,cv2.CV_16S,0,1)

# absX = cv2.convertScaleAbs(x)   # 转回uint8
# absY = cv2.convertScaleAbs(y)      
# dst = cv2.addWeighted(absX,1,absY,1,0)

# plt.figure()
# plt.subplot(121)
# plt.imshow(dst,'gray')
# plt.title('sobel img')

# dst1 = 255 - gray
# plt.subplot(122)
# plt.imshow(dst1,'gray')
# plt.title('reverse sobel img')

# ret,binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
# draw_img11 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
# plt.imshow(draw_img11)


# side_threshold = gray.max()*0.8
# ret2,thresh_2 = cv2.threshold(gray,side_threshold,255,2)
# plt.figure()
# plt.imshow(thresh_2,'gray')


#%%
# 3. 定義細胞核的"重心"為中點(A)
# """
# ————————————————
# 版权声明：本文为CSDN博主「DL数据集」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_42216109/article/details/89840323

# """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY)
    
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    new_contours = []
    new_hierarchy =[]
    for i in range(len(contours)):
        if len(contours[i]) >=10:
            new_contours.append(contours[i])
            new_hierarchy.append(hierarchy[0][i])
    print(f"new_contours = {len(new_contours)}")
    
    new_contours_area=[]
    if len(new_contours)>1:
        for i in range(len(new_contours)):
            new_contours_area.append(int(cv2.contourArea(new_contours[i])))
    else :
        new_contours_area.append(int(cv2.contourArea(new_contours[0])))
       
    max_area = int(max(new_contours_area))
    loc = new_contours_area.index(max_area)
    correct__contours  = new_contours[loc]

    locals()['draw_img'+str(loc)] = cv2.drawContours(img.copy(),correct__contours,-1,(0,255,255),3)
    # cv2.imshow('draw_img'+str(loc), locals()['draw_img'+str(loc)])
    # plt.figure()
    plt.subplot(3,1,3)
    plt.imshow(locals()['draw_img'+str(loc)])
    
    
    mom = cv2.moments(correct__contours)
    pt = (int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00']))  # 使用前三个矩m00, m01和m10计算重心
    # cv2.circle(draw_img1, pt, 2, (0, 0, 255), 2)  # 画红点
    print(f"pt = {pt}")
    
    Center_to_edge=[]
    for i in correct__contours:
        result = ((pt[0]-i[0][0])**2 + (pt[1]-i[0][1])**2)**0.5
        Center_to_edge.append(result)
    print(f"Center_to_edge 的個數 = {len(Center_to_edge)}")

# 5. 計算這陣列裡的"中位數"，即為細胞質的半徑(R)

    def mediannum(num):
        listnum = [num[i] for i in range(len(num))]
        listnum.sort()
        lnum = len(num)
        if lnum % 2 == 1:
            i = int((lnum + 1) / 2)-1
            return listnum[i]
        else:
            i = int(lnum / 2)-1
            return (listnum[i] + listnum[i + 1]) / 2
        
    med = mediannum(Center_to_edge)
    print(f"中位數 = {med}")
    
    area = cv2.contourArea(correct__contours)  
    nc_ratio = area/(math.pi*med**2)
    print(f"核質比 = {nc_ratio}")
    print(f"重心 = {pt}")
    print(f"細胞核面積 = {area}")

def main():
    otsu(r'C:\Users\Jason\Desktop\otsu\Normal\C20-18-2-9.PNG')
# except:
#     print(f"找不到輪廓")
if __name__ == '__main__':
    main()
    
    #%%
    # for i in range(0,len(new_contours)):
    #     locals()['draw_img'+str(i)] = cv2.drawContours(img.copy(),new_contours,i,(0,255,255),3)
    #     cv2.imshow('draw_img'+str(i), locals()['draw_img'+str(i)])
        # plt.figure()
        # plt.imshow(locals()['draw_img'+str(i)])
    
    
    # for i in range(len(new_contours)):
    #     mom = cv2.moments(new_contours[i])
    #     pt = (int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00']))  # 使用前三个矩m00, m01和m10计算重心
    #     # cv2.circle(draw_img1, pt, 2, (0, 0, 255), 2)  # 画红点
    #     print(f"pt = {pt}")
        # cv2.putText(draw_img1, text, (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, 8, 0);
    
    # cv2.imshow("draw_img1", draw_img1)



# ————————————————
# 版权声明：本文为CSDN博主「iracer」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/iracer/article/details/90695914


#%%
# 4. 計算(A)與每一個細胞質邊的像素(2.)，計算其向量長度，並將每一點的向量長度放在一陣列中。
    # Center_to_edge=[]
    # for i in new_contours:
    #     result = ((pt[0]-i[0][0][0])**2 + (pt[1]-i[0][0][1])**2)**0.5
    #     Center_to_edge.append(result)
    # print(len(Center_to_edge))
    
    # # 5. 計算這陣列裡的"中位數"，即為細胞質的半徑(R)
    
    # def mediannum(num):
    #     listnum = [num[i] for i in range(len(num))]
    #     listnum.sort()
    #     lnum = len(num)
    #     if lnum % 2 == 1:
    #         i = int((lnum + 1) / 2)-1
    #         return listnum[i]
    #     else:
    #         i = int(lnum / 2)-1
    #         return (listnum[i] + listnum[i + 1]) / 2
        
    # med = mediannum(Center_to_edge)
    # print(f"中位數 = {med}")
# ————————————————
# 版权声明：本文为CSDN博主「清风不识字12138」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_33363973/article/details/78773144

#%%

# 6. 核質比 = (細胞核面積)/(pi*R^2)
#     area = cv2.contourArea(contours[1])  
#     nc_ratio = area/(math.pi*med**2)
#     print(f"核質比 = {nc_ratio}")
#     print(f"重心 = {pt}")
#     print(f"細胞核面積 = {area}")


# except:
#     print(f"找不到輪廓")


#%%

# contours, hierarchy = cv2.findContours(th1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# img1 = cv2.drawContours(gray, contours, -1, (0, 0, 255), 2)
# binaryIMG = cv2.Canny(img1, 16, 90)
# plt.imshow(binaryIMG)

# cv2.imshow('',binaryIMG)



#參考用
# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(gray,(3,3),0)
#     detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)
 
# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3
 
# img = cv2.imread(r"C:\Users\Jason\Desktop\otsu\Normal\C20-18-1-7.PNG")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# cv2.namedWindow('canny demo')
 
# cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
 
# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()


# import cv2

# img = cv2.imread(r"C:\Users\Jason\Desktop\otsu\Normal\C20-18-1-7.PNG")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
# draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
# draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
# draw_img3 = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)

# x, y, w, h = cv2.boundingRect(img)


# print ("contours:类型：",type(contours))
# print ("第0 个contours:",type(contours[0]))
# print ("contours 数量：",len(contours))

# print ("contours[0]点的个数：",len(contours[0]))
# print ("contours[1]点的个数：",len(contours[1]))

# cv2.imshow("img", img)
# cv2.imshow("draw_img0", draw_img0)
# cv2.imshow("draw_img1", draw_img1)
# cv2.imshow("draw_img2", draw_img2)
# cv2.imshow("draw_img3", draw_img3)
# ————————————————
# 版权声明：本文为CSDN博主「SongpingWang」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/wsp_1138886114/article/details/82945328
