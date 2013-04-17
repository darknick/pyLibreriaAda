import cv2.cv as cv
import cv2
import tesseract
import math
import numpy as np
import os
import sys

## Comprobacion de Sistema Operativo
sistemaop = sys.platform
if sistemaop=='darwin':
	print 'Estas usando Mac'
elif sistemaop=='win32' or sistemaop=='win64':
	print 'Estas en Win'
else:
	print 'No estas ni en mac ni en win'
####################################

## Carga de archivos
# List files in "albaranes" folder
ficheros = os.listdir('albaranes') 
print ficheros
# Open one file
img = cv2.imread("albaranes/"+ficheros[0])
####################################


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,100,255,1)

# Remove some small noise if any.
dilate = cv2.dilate(thresh,None)
erode = cv2.erode(dilate,None)

# Find contours with cv2.RETR_CCOMP
contours,hierarchy = cv2.findContours(erode,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i,cnt in enumerate(contours):

    # Check if it is an external contour and its area is more than 1500 CHILDREN
    if hierarchy[0,i,2] == -1 and cv2.contourArea(cnt)>1500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

        m = cv2.moments(cnt)
        cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
        cv2.circle(img,(int(cx),int(cy)),3,255,-1)

    # Check if it is an external contour and its area is more than 2000 PARENT
    if hierarchy[0,i,3] == -1 and cv2.contourArea(cnt)>2000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)

        m = cv2.moments(cnt)
        cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
        cv2.circle(img,(int(cx),int(cy)),3,255,-1)
    

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()