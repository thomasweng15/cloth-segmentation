import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage import morphology
import os

"""
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle.
ix,iy = -1,-1
def draw_rectangle(event,x,y,flags,param):
	global ix,iy,drawing,mode,img_rgb,i
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			img_rgb = cv2.rectangle(img_rgb,(ix,iy),(x,y),(0,255,0),2)
			print(ix,iy)
			print(x,y)
			##np.save("../shirt_val_bdbox/"+str(i+316)+"bdbox", np.array([ix,iy,x,y]))
			cv2.imshow('img_rgb', img_rgb)
		else:
			cv2.circle(img_rgb,(x,y),5,(0,0,255),-1)

row_start = 380
row_end = 1276
col_start = 0
col_end = -1
step = 4

img_rgb = cv2.imread("/Users/Aurora/Desktop/data_towel/tshirt-inner/camera_0/0_16_12_2019_22:29:56.jpg")

cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img_rgb',draw_rectangle)
while 1:
	cv2.imshow('img_rgb', img_rgb)
	key = cv2.waitKey(0) & 0xFF

	if key == ord("r"):
		img_rgb = clone.copy()
	elif key == ord(" "):
		break

"""
#256 316
#790 716
def index(x):
		return(int(x.split("_")[1]))

dire = "/media/ExtraDrive2/jianingq/data_studio_painted_towel_r0.1/camera_3/"#data_painted_towel_table/camera_3/"
filename =  os.listdir(dire)
#filename = sorted(filename, key = index) 
jpgs = [x for x in filename if (x.endswith(".png") and x.startswith("rgb"))]
pcds = [x for x in filename if x.endswith(".npy")]
for i in [1000]:#[]range(1):#len(jpgs)):
	print(i)
	print(jpgs[i])
	img_rgb = cv2.imread(dire+jpgs[i])
	#row_start = 150
	#row_end = 660
	#col_start = 215
	#col_end = 1100
	img_rgb = img_rgb#[row_start:row_end, col_start:col_end, :]
	img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
	h, w, _ = img_rgb.shape

# --------------------------------------------

	# Range for lower red
	lower_red = np.array([0,175,70])
	upper_red = np.array([7,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
	 
	# Range for upper red
	lower_red = np.array([170,150,70])
	upper_red = np.array([180,255,255])
	mask2 = cv2.inRange(img_hsv,lower_red,upper_red)
	 
	# Generating the final mask to detect red color
	labels_red = mask1 + mask2
 
 # -------------------------------------------

	#defining the range of Yellow color
	yellow_lower = np.array([20,100,70],np.uint8)
	yellow_upper = np.array([31,255,255],np.uint8)
	labels_yellow = cv2.inRange(img_hsv, yellow_lower, yellow_upper)

# --------------------------------------------

	#defining the range of purple color
	purple_lower = np.array([110,120,70],np.uint8)
	purple_upper = np.array([130,255,255],np.uint8)
	labels_purple = cv2.inRange(img_hsv, purple_lower, purple_upper)

# --------------------------------------------

	#defining the range of blue color
	blue_lower = np.array([100,120,70],np.uint8)
	blue_upper = np.array([110,255,255],np.uint8)
	labels_blue = cv2.inRange(img_hsv, blue_lower, blue_upper)

# --------------------------------------------

	#defining the range of green color
	green_lower = np.array([31,50,70],np.uint8)
	green_upper = np.array([100,255,255],np.uint8)
	labels_green = cv2.inRange(img_hsv, green_lower, green_upper)

# --------------------------------------------

	# Range for orange
	lower_orange = np.array([10, 100, 20])
	upper_orange = np.array([20,255,255])
	 
	# Generating the final mask to detect red color
	labels_orange = cv2.inRange(img_hsv,lower_orange,upper_orange)

# --------------------------------------------	
	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	labels_red = cv2.morphologyEx(labels_red, cv2.MORPH_CLOSE, kernel)
	labels_red = cv2.morphologyEx(labels_red, cv2.MORPH_OPEN, kernel)
	labels_red = morphology.remove_small_objects(labels_red>0, min_size=100, connectivity=1)
	labels_red = labels_red * 255

	labels_yellow = cv2.morphologyEx(labels_yellow, cv2.MORPH_OPEN, kernel)
	labels_yellow = cv2.morphologyEx(labels_yellow, cv2.MORPH_CLOSE, kernel)
	labels_yellow = morphology.remove_small_objects(labels_yellow>0, min_size=100, connectivity=1)
	labels_yellow = labels_yellow * 255

	labels_purple = cv2.morphologyEx(labels_purple, cv2.MORPH_OPEN, kernel)
	labels_purple = cv2.morphologyEx(labels_purple, cv2.MORPH_CLOSE, kernel)
	labels_purple = morphology.remove_small_objects(labels_purple>0, min_size=100, connectivity=1)
	labels_purple = labels_purple * 255

	labels_blue = cv2.morphologyEx(labels_blue, cv2.MORPH_OPEN, kernel)
	labels_blue = cv2.morphologyEx(labels_blue, cv2.MORPH_CLOSE, kernel)
	labels_blue = morphology.remove_small_objects(labels_blue>0, min_size=100, connectivity=1)
	labels_blue = labels_blue * 255

	labels_green = cv2.morphologyEx(labels_green, cv2.MORPH_OPEN, kernel)
	labels_green = cv2.morphologyEx(labels_green, cv2.MORPH_CLOSE, kernel)
	labels_green = morphology.remove_small_objects(labels_green>0, min_size=100, connectivity=1)
	labels_green = labels_green * 255

	labels_orange = cv2.morphologyEx(labels_orange, cv2.MORPH_OPEN, kernel)
	labels_orange = cv2.morphologyEx(labels_orange, cv2.MORPH_CLOSE, kernel)
	labels_orange = morphology.remove_small_objects(labels_orange>0, min_size=100, connectivity=1)
	labels_orange = labels_orange * 255




	#cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
	#cv2.imshow('rgb', img_rgb)
	#cv2.namedWindow('labels_red', cv2.WINDOW_NORMAL)
	#cv2.imshow('labels_red', (img_rgb*labels_red[:,:,None]).astype(np.uint8))
	#cv2.namedWindow('labels_yellow', cv2.WINDOW_NORMAL)
	#cv2.imshow('labels_yellow', labels_yellow.astype(np.uint8))
	#cv2.namedWindow('labels_purple', cv2.WINDOW_NORMAL)
	#cv2.imshow('labels_purple', labels_purple.astype(np.uint8))
	#cv2.namedWindow('labels_blue', cv2.WINDOW_NORMAL)
	#cv2.imshow('labels_blue', (img_rgb*labels_blue[:,:,None]).astype(np.uint8))
	#cv2.namedWindow('labels_green', cv2.WINDOW_NORMAL)
	#cv2.imshow('labels_green', (img_rgb*labels_green[:,:,None]).astype(np.uint8))
	#cv2.namedWindow('labels_orange', cv2.WINDOW_NORMAL)
	#cv2.imshow('labels_orange', labels_orange.astype(np.uint8))

	cv2.imwrite('./test_color/img_red.png', (img_rgb*labels_red[:,:,None]).astype(np.uint8))
	cv2.imwrite('./test_color/img_yellow.png',(img_rgb*labels_yellow[:,:,None]).astype(np.uint8))
	cv2.imwrite('./test_color/img_green.png',(img_rgb*labels_green[:,:,None]).astype(np.uint8))
	cv2.imwrite('./test_color/img_rgb.png',(img_rgb).astype(np.uint8))

	#cv2.imwrite(dire+(jpgs[i].split("_")[1])+'_labels_red.png', labels_red)
	#print(dire+(jpgs[i].split("_")[1])+'_labels_red.png')
	#cv2.imwrite(dire+(jpgs[i].split("_")[1])+'_labels_yellow.png',labels_yellow)
	#cv2.imwrite(dire+(jpgs[i].split("_")[1])+'_labels_purple.png', labels_purple)
	#cv2.imwrite(dire+(jpgs[i].split("_")[1])+'_labels_blue.png', labels_blue)
	#cv2.imwrite(dire+(jpgs[i].split("_")[1])+'_labels_green.png', labels_green)
	#print(dire+(jpgs[i].split("_")[1])+'_labels_green.png')
	#cv2.imwrite(dire+(jpgs[i].split("_")[1])+'_labels_orange.png', labels_orange)

	
	# cv2.imwrite("/scratch/luxinz/towel_video_train/" + str(i) + '/yellow_labels.png', labels_yellow)

	#cv2.imwrite("/scratch/luxinz/shirt_video_train/" + str(i) + '/corner_labels.png', labels_red)
	#cv2.imwrite("/scratch/luxinz/shirt_video_train/" + str(i) + '/edge_labels.png', labels_yellow+labels_blue+labels_purple+labels_green)
	#cv2.imwrite("/scratch/luxinz/shirt_video_train/" + str(i) + '/sleeve_labels.png', labels_blue)
	#cv2.imwrite("/scratch/luxinz/shirt_video_train/" + str(i) + '/shoulder_labels.png', labels_purple)
	#cv2.imwrite("/scratch/luxinz/shirt_video_train/" + str(i) + '/collar_labels.png', labels_green)

	#cv2.waitKey(0)


