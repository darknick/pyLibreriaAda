import tesseract
import cv2
import cv2.cv as cv

ruta = "albaranestiff/alb_0001.tiff"
image0=cv2.imread(ruta)
print "Cargando %s"%ruta
#### you may need to thicken the border in order to make tesseract feel happy to ocr your image #####
offset=20
height,width,channel = image0.shape
image1=cv2.copyMakeBorder(image0,offset,offset,offset,offset,cv2.BORDER_CONSTANT,value=(255,255,255)) 
# cv2.namedWindow("Test")
# cv2.imshow("Test", image1)
# cv2.waitKey(0)
# cv2.destroyWindow("Test")
#####################################################################################################
api = tesseract.TessBaseAPI()
api.Init(".","tmp3",tesseract.OEM_DEFAULT)
#############################################################
#api.SetVariable("global_tessdata_manager_debug_level","True")		#Increase verbosity (Debug)
api.SetVariable("global_load_punc_dawg","False")					#Ignore punctuation patterns
#api.SetVariable("global_load_number_dawg","False")					#Ignore number patterns
api.SetVariable("language_model_penalty_non_freq_dict_word","0.2") 	#Penalty for words not in the frequent word dictionary(0.1 Default)
#api.SetVariable()
#api.SetVariable()
#api.SetVariable()
api.SetPageSegMode(tesseract.PSM_OSD_ONLY)
"""
PSM_OSD_ONLY 				Orientation and script detection only.
PSM_AUTO_OSD 				Automatic page segmentation with orientation and script detection. (OSD)
PSM_AUTO_ONLY 				Automatic page segmentation, but no OSD, or OCR.
PSM_AUTO 					Fully automatic page segmentation, but no OSD.
PSM_SINGLE_COLUMN 			Assume a single column of text of variable sizes.
PSM_SINGLE_BLOCK_VERT_TEXT 	Assume a single uniform block of vertically aligned text.
PSM_SINGLE_BLOCK 			Assume a single uniform block of text. (Default.)
PSM_SINGLE_LINE 			Treat the image as a single text line.
PSM_SINGLE_WORD 			Treat the image as a single word.
PSM_CIRCLE_WORD 			Treat the image as a single word in a circle.
PSM_SINGLE_CHAR 			Treat the image as a single character.
PSM_COUNT 					Number of enum entries.
"""

height1,width1,channel1=image1.shape
print image1.shape
print image1.dtype.itemsize
width_step = width*image1.dtype.itemsize
print width_step
#method 2
cvmat_image=cv.fromarray(image1)
iplimage =cv.GetImage(cvmat_image)
print iplimage

tesseract.SetCvImage(iplimage,api)
#api.SetImage(m_any,width,height,channel1)
text=api.GetUTF8Text()
conf=api.MeanTextConf()
image=None
print "..............."
print "Ocred Text: \n %s"%text
print "Cofidence Level: %d %%"%conf
print "\n Conf:\n %s"%conf

raw_input("Press Enter to continue...")
