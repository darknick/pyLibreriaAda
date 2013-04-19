import tesseract
import cv2
import cv2.cv as cv
import os
import re #RegEx
import pprint #Debug

## Carga de archivos
folder = "albaranes2"
# List files in "albaranes" folder
ficheros = os.listdir(folder) 

print ficheros
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
api.SetPageSegMode(tesseract.PSM_SINGLE_BLOCK)
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
arrcol = [
	'cantidad',
	'nombre',
	'aparicion',
	'codigo-titulo']

datos = []
dato = []
for col,indice in enumerate(ficheros):
	ruta = folder+"/"+ficheros[col]
	image1=cv2.imread(ruta)
	print "Cargando %s"%ruta
	"""
	#### you may need to thicken the border in order to make tesseract feel happy to ocr your image #####
	offset=20
	height,width,channel = image0.shape
	image1=cv2.copyMakeBorder(image0,offset,offset,offset,offset,cv2.BORDER_CONSTANT,value=(255,255,255)) 
	# cv2.namedWindow("Test")
	# cv2.imshow("Test", image1)
	# cv2.waitKey(0)
	# cv2.destroyWindow("Test")
	#####################################################################################################
	"""

	height1,width1,channel1=image1.shape
	print image1.shape
	print image1.dtype.itemsize
	width_step = width1*image1.dtype.itemsize
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
	api.Clear()
	##Procesamiento de texto
	#Eliminar ciertos caracteres

	print "...............\nTexto sin procesar:\n\n"+text
	done = 0
	text = text.replace('\n\n','\n') 	#Clean white lines
	text = text.replace('l','1') 		#Misses so often...
	text = text.upper()

	if col==0:
		text = text.replace('\n', ';')
		done = 1

	elif col==1:
		text = text.replace('\n', ';')
		done = 1

	elif col==2:
		text = text.replace('\n', ';')
		done = 1

	elif col==3:
		text = text.replace('O', '0')
		text = re.sub(r'[^\d\n ]','',text)
		text = text.replace(' 0 ',',')
		text = re.sub('\n ?',';',text)
		done = 1
	dato = text.split(';')
	datos.append(dato)
"""
	if done:
		print "..............."
		print arrcol[col]+" procesado:\n\n%s\n\n"%text
#	print "Cofidence Level: %d %%"%conf
"""
print '\n\n'
for i in datos:
		print i
raw_input("Press Enter to continue...")

"""
Cargamos lista de imagenes
Bucle de Procesamiento
	Segmentado
	(?)Busqueda de huecos en segmento
		Dividir imagen
		Calcular huecos
	o
		Buscar zona blanca
			Cuadrado de anchura ~= segmento

	Guardar imagen segmentos
	Array punteros

	Inicializacion OCR
	Bucle ocr en array de segmentos
		Ejecuta ocr sobre segmento
		Procesado salida ocr (regex, limpieza, etc)
			Segun segmento

		Si conocemos los huecos sera mas facil
		Comprobaciones






"""