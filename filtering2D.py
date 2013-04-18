
import cv2
import numpy as np

img = cv2.imread('lena.jpg')

kernel = np.array([ [0,-1,0],
                    [-1,5,-1],
                    [0,-1,0] ],np.float32)   # kernel should be floating point type.

new_img = cv2.filter2D(img,-1,kernel)        # ddepth = -1, means destination image has depth same as input image.

cv2.imshow('img',img)
cv2.imshow('new',new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()