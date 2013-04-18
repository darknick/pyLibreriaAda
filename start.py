import cv2.cv as cv
import cv2
import tesseract
import math
import numpy as np
import os
import sys

## Funciones

def show2Images(img1,img2):
	##Show 2 images 1 window
	imgc = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	imgc2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	#Suma
	both = np.hstack((imgc2,imgc))
	#Muestra
	cv2.imshow('imgc',both)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
##############################################
def recognizeText(img):
	api = tesseract.TessBaseAPI()
	api.Init(".","spa",tesseract.OEM_DEFAULT)
	#api.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyz")
	api.SetPageSegMode(tesseract.PSM_AUTO)
	height,width,channel=img.shape
	#print img.shape
	#print img.dtype.itemsize
	#width_step = width*img.dtype.itemsize
	#print width_step
	#method 1 
	iplimage = cv.CreateImageHeader((width,height), cv.IPL_DEPTH_8U, channel)
	cv.SetData(iplimage, img.tostring(),img.dtype.itemsize * channel * (width))
	tesseract.SetCvImage(iplimage,api)
	text=api.GetUTF8Text()
	conf=api.MeanTextConf()
	image=None
	#print "..............."
	print "Revistas: %s"
	print text
	#print "Cofidence Level: %d %%"%conf
	#cv2.waitKey(1000)

##############################################





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
folder = "albaranes"
# List files in "albaranes" folder
ficheros = os.listdir(folder) 

print ficheros
# Open one file
for i,fichero in enumerate(ficheros):
	img = cv2.imread(folder+"/"+fichero)
	height,width,channel=img.shape
	####################################

	##Procesado
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#thresholg for OCR
	ret2,thresh2 = cv2.threshold(gray,100,255,1)
	#threshold for detection contour
	ret,thresh = cv2.threshold(gray,100,255,1)
	#Create an erode element for operation
	erosion_x = 1
	erosion_y = 1
	erosion_size_x = 3
	erosion_size_y = 3
	erosion_type = cv2.MORPH_ELLIPSE
	element = cv2.getStructuringElement(erosion_type,
		( erosion_size_x + erosion_x, erosion_size_y + erosion_y ),
		( erosion_x, erosion_y ) 
		)
	# Remove some small noise if any.
	dilate = cv2.dilate(thresh,element)
	erode = cv2.erode(dilate,element)
	#Showing images
	#show2Images(dilate,erode)

	# Find contours with cv2.RETR_CCOMP
	contours,hierarchy = cv2.findContours(erode,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

	parents=[]
	parents_cnt=[]
	childrens=[]
	childrens_cnt=[]

	for i,cnt in enumerate(contours):

	    # Check if it is an external contour and its area is more than 2000 PARENT
		if hierarchy[0,i,3] == -1 and cv2.contourArea(cnt)>20000:
			(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
			m = cv2.moments(cnt)
			cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
			#cv2.circle(img,(int(cx),int(cy)),3,255,-1)
			#print angle
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

			mask = np.zeros(gray.shape,np.uint8)
			cv2.drawContours(mask,[cnt],0,200,-1)
			res= cv2.bitwise_and(thresh2,thresh2,mask=mask)
			#Rotate
			#image_center = tuple(np.array(res.shape)/2)
			#calculo del angulo
			#angle2 = 90-angle
			#rot_mat = cv2.getRotationMatrix2D((cx,cy),angle+angle2,1.0)
			#res = cv2.warpAffine(res, rot_mat, res.shape,flags=cv2.INTER_LINEAR)
			#mean = cv2.mean(img,mask = mask)
			res = cv2.merge([res,res,res],None)
			parents.append(res)
			#dst = cv2.resize(res,dsize=(900,900),interpolation=cv.CV_INTER_LINEAR)
			#cv2.imshow('img',dst)
			#cv2.waitKey(0)

	#for i,parent in enumerate(parents):
	recognizeText(parents[0])

	cv2.destroyAllWindows()

	
	

"""
	# Check if it is an external contour and its area is more than 900 CHILDREN
	if hierarchy[0,i,2] == -1 and cv2.contourArea(cnt)>900:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
		children.append(cnt) 

		mask = np.zeros(gray.shape,np.uint8)
		cv2.drawContours(mask,[cnt],0,255,-1)
		mean = cv2.mean(img,mask = mask)
		dst = cv2.resize(img,dsize=(900,900),interpolation=cv.CV_INTER_LINEAR)
		cv2.imshow('img',dst)
		cv2.waitKey(0)
"""








