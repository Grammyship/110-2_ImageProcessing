""" 
110-2 NTNU CSIE
ImageProcessing
Assignment4
by 40847041S 朱自宇
"""

import numpy as np
import cv2
import matplotlib as plt

# Load the image
img = cv2.imread("me.png")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# do the average filter
averageFilter = cv2.blur(grayImg, (3,3))          
                
# do the median filter
medianFilter = cv2.medianBlur(grayImg, 3)

# do the unsharp masking
k = 0.6
averageImg = ( grayImg - k*averageFilter ) / ( 1 - k )
averageImg = averageImg.astype(np.uint8)
medianImg = ( grayImg - k*medianFilter ) / ( 1 - k )
medianImg = medianImg.astype(np.uint8)

# show the results
cv2.imshow("origin picture", grayImg)
cv2.imshow("average filter", averageImg)
cv2.imshow("median filter", medianImg)


# wait until the user press any button to close the img window
cv2.waitKey(0)
cv2.destroyAllWindows()