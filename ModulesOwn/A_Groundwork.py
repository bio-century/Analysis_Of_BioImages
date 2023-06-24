# Color DNA with respect to their bases
def identifyCellContoursAndAreaOverlays(myImage, myImageThresholded, dictThresholdValues):
	# sources
	# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
	# https://de.mathworks.com/matlabcentral/answers/43435-i-couldn-t-understand-convex-area
	# https://stackoverflow.com/questions/32401806/get-mask-from-contour-with-opencv
	# https://stackoverflow.com/questions/50591442/convert-3-dim-image-array-to-2-dim
	# https://www.tutorialspoint.com/how-to-compute-the-area-and-perimeter-of-an-image-contour-using-opencv-python

	# import modules
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.measure import regionprops
	from statistics import mean
	import math

	cellCount           = 0
	contours, hierarchy = cv2.findContours(myImageThresholded.astype(np.uint8), 1, 2)
	myMaskContoursAll   = np.zeros(myImage.shape)
	numberOfObjects     = len(contours)
	list_multicore      = []
	oneNucleiArea      	= []

	for ii, cnt in enumerate(contours):
		M               = cv2.moments(cnt)
		area            = cv2.contourArea(cnt)
		perimeter       = cv2.arcLength(cnt, True)
		perimeter       = round(perimeter, 1)
		myImageContours = cv2.drawContours(myImage, [cnt], -1, (0, 0, 255), 2)
		
		if M['m00'] != 0.0:
			x1 = int(M['m10'] / M['m00'])
			y1 = int(M['m01'] / M['m00'])
			x1_rounded = round(x1)
			y1_rounded = round(y1)
			myImageContours[y1_rounded - 2 : y1_rounded + 2, x1_rounded - 2 : x1_rounded + 2] = (0, 0, 255)
		cv2.putText(myImageContours, f'{ii + 1}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

		myMaskZeros       = np.zeros(myImage.shape, np.uint8)
		myMaskContoursTmp = cv2.drawContours(image = myMaskZeros, contours=[cnt], contourIdx = -1, color = (255, 255, 255), thickness = cv2.FILLED)

		myMaskContours                           = np.zeros(myImage.shape)
		myMaskContours[myMaskContoursTmp == 255] = 1

		props = regionprops(myMaskContoursTmp, cache = False)
		prop = props[0]
		
		ratio_conv_filled = prop.convex_area / prop.filled_area
		# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
		# prop.convex_area / prop.filled_area
		# convex_area (convex hull): the smallest region that satisfy two conditions: (1) it is convex (2) it contains the original region.
		# => The ratio increases the less convex the area is (multiple nuclei)
		# celladd = math.ceil(prop.filled_area)/(2291.9454545454546)
		# celladd = math.ceil((prop.filled_area)/(2276.51))
		celladd = round((prop.filled_area)/(2276.51))
		# print(celladd)

		if  ratio_conv_filled > dictThresholdValues["3CellCluster"]:
			myMaskContoursAll  = myMaskContoursAll + myMaskContours * 0.9
			list_multicore.append(1)
			# celladd = 3
			cellCount=cellCount + celladd
		elif  ratio_conv_filled > dictThresholdValues["2CellCluster"]:
			myMaskContoursAll  = myMaskContoursAll + myMaskContours * 0.5
			list_multicore.append(1)
			# celladd = 2
			cellCount = cellCount + celladd
		else:
			myMaskContoursAll  = myMaskContoursAll + myMaskContours * 0.2
			list_multicore.append(0)
			# celladd = 1
			cellCount = cellCount + celladd
			oneNucleiArea.append(prop.filled_area)
		myMaskContoursAll = cv2.drawContours(myMaskContoursAll, [cnt], -1, (0,0,255), 2)
		cv2.putText(myMaskContoursAll, f'{ii + 1}', (x1, y1 + 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (150, 150, 150), 1)
		cv2.putText(myMaskContoursAll, "%.3f" % ratio_conv_filled, (x1, y1+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 0, 0), 1)
	list_avg = mean(oneNucleiArea) 
	# print(list_avg)
	return cellCount, contours, myImageContours, myMaskContoursAll