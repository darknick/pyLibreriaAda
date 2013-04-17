import cv2.cv as cv
import cv2
import tesseract
import math
import numpy
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

##Load Images
#list files in "Albaranes"
ficheros = os.listdir('albaranes') 
print ficheros
#Open one file
image0 = cv2.imread("albaranes/"+ficheros[0], cv.CV_LOAD_IMAGE_COLOR)
#Make border to feel happy with OCR
offset=20
height,width,channel = image0.shape
image1=cv2.copyMakeBorder(image0,offset,offset,offset,offset,cv2.BORDER_CONSTANT,value=(255,255,255)) 
#cv2.namedWindow("Test")
#cv2.imshow("Test", image1)
#cv2.waitKey(0)
#cv2.destroyWindow("Test")
######################################

##Procesado
#Covert to grayscale
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
print type(gray)
cvmat_image=cv.fromarray(gray)
cv.Smooth(cvmat_image,cvmat_image,smoothtype=cv.CV_GAUSSIAN, param1=3, param2=3, param3=0, param4=0)
print type(cvmat_image)
graynew = numpy.asarray(cvmat_image)



#Canny detector
edges = cv2.Canny(graynew, 80, 120)

cv2.namedWindow("Test")
cv2.imshow("Test", edges)
cv2.waitKey(0)
cv2.destroyWindow("Test")

#Hough transform
lines = cv.HoughLines2(edges, 1, math.pi/2, 2, None, 30, 1);
for line in lines[0]:
    pt1 = (line[0],line[1])
    pt2 = (line[2],line[3])
    cv2.line(image1, pt1, pt2, (0,0,255), 3)

cv2.namedWindow("Test")
cv2.imshow("Test", image1)
cv2.waitKey(0)
cv2.destroyWindow("Test")

