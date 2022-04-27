import numpy as np
import cv2

# read the image
img = cv2.imread("takagisan.png", cv2.IMREAD_COLOR)

# print out the original picture size
# print(img.shape)

"""
The original version of the picture
"""

# show the picture
cv2.imshow("Original", img)


"""
now do the grayscale
"""

# set the gray image variable
grayimg = np.zeros( (img.shape[0],img.shape[1]) , np.uint8 )

# divide the BGR of the original image by 3
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        (b,g,r) = img[i,j]
        gray = (int(b) + int(g) + int(r))/3
        grayimg[i,j] = np.uint8(gray)

# show the grayscale version of the picture
cv2.imshow("Gray", grayimg)

# save the picture
cv2.imwrite("gray_takagisan.png",grayimg)

# wait until the user press any button to close the img window
cv2.waitKey(0)
cv2.destroyAllWindows()