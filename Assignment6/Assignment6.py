""" 
110-2 NTNU CSIE
ImageProcessing
Assignment6
by 40847041S 朱自宇
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# read the gray image
grayImg = cv2.imread("Ukraine.png", cv2.IMREAD_GRAYSCALE)

# gray scale count of the image
grayHE = np.zeros(256,dtype=np.uint32)
for i in grayImg:
    for j in i:
        grayHE[j]+=1
        
# total size of image
ImgSize = grayImg.shape[0]*grayImg.shape[1]

# record the Max t
OtsuMax = 0
t = 0
temp = 0

# calculate m
m = 0
for i in range(256):
    m += i*grayHE[i]
m = m / ImgSize

# find t
for i in range(256):
    a_t = 0
    b_t = 0
    m_a = 0
    for j in range(i+1):
        a_t += grayHE[j]
        m_a += j*grayHE[j]
    b_t = (ImgSize - a_t)/ImgSize
    a_t = a_t / ImgSize
    m_a = m_a / ImgSize
    if a_t != 0 and b_t != 0:
        temp = ((m_a - m*a_t)**2)/(a_t*b_t)
    else: temp = 0
    if OtsuMax < temp:
        OtsuMax = temp
        t = i

grayImg_flat = grayImg.flatten()
output = np.zeros(grayImg_flat.size,dtype=np.uint8)
for i in range(grayImg_flat.size):
    if grayImg_flat[i] < t:
        output[i] = 0
    else: output[i] = 255

output = output.reshape( (grayImg.shape[0],grayImg.shape[1]) )


# show the original gray image
cv2.imshow("Input Gray image", grayImg)

# show the output gray image
cv2.imshow("Output Otsu threshold image", output)

# wait until the user press any button to close the img window
cv2.waitKey(0)
cv2.destroyAllWindows()