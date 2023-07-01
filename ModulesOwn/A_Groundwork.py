def ellipseFit(myImage, myImageThresholded):
	# sources of main algorithms
	# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
	# User: Costas Voglis
	# and
	# https://stackoverflow.com/questions/62698756/opencv-calculating-orientation-angle-of-major-and-minor-axis-of-ellipse
	# User: fmw42, Fred Weinhaus

	# import modules
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.measure import regionprops
	import math

	cellCount           = 0
	contours, hierarchy = cv2.findContours(myImageThresholded.astype(np.uint8), 1, 2)
	myMaskContoursAll   = np.zeros(myImage.shape)
	numberOfObjects     = len(contours)
	list_multicore      = []

	for ii, cnt in enumerate(contours):
		M               = cv2.moments(cnt)
		area            = cv2.contourArea(cnt)
		perimeter       = cv2.arcLength(cnt, True)
		perimeter       = round(perimeter, 1)
		
		# compute center of mass
		# see sources of main algorithms
		if M['m00'] != 0.0:
			x1 = int(M['m10'] / M['m00'])
			y1 = int(M['m01'] / M['m00'])
			x1_rounded = round(x1)
			y1_rounded = round(y1)
		myMaskZeros       = np.zeros(myImage.shape, np.uint8)
		myMaskContoursTmp = cv2.drawContours(image = myMaskZeros, contours=[cnt], contourIdx = -1, color = (255, 255, 255), thickness = cv2.FILLED)
		myMaskContours                           = np.zeros(myImage.shape)
		myMaskContours[myMaskContoursTmp == 255] = 1
		props = regionprops(myMaskContoursTmp, cache = False)
		prop = props[0]


		# compute ellipse fit incl. major and minor axis
		# see sources of main algorithms
		ellipse = cv2.fitEllipse(cnt)
		(xc, yc), (d1, d2), angle = ellipse
		cv2.ellipse(myImage, ellipse, (0, 0, 170), 2)
		rmajor = max(d1, d2) / 2
		if angle > 90:
			angle = angle - 90
		else:
			angle = angle + 90
		x1 = xc + math.cos(math.radians(angle)) * rmajor
		y1 = yc + math.sin(math.radians(angle)) * rmajor
		x2 = xc + math.cos(math.radians(angle + 180)) * rmajor
		y2 = yc + math.sin(math.radians(angle + 180)) * rmajor
		cv2.line(myImage, (int(x1), int(y1)), (int(x2), int(y2)), (170, 0, 0), 2)

		rminor = min(d1, d2) / 2
		if angle > 90:
			angle = angle - 90
		else:
			angle = angle + 90

		x1 = xc + math.cos(math.radians(angle)) * rminor
		y1 = yc + math.sin(math.radians(angle)) * rminor
		x2 = xc + math.cos(math.radians(angle + 180)) * rminor
		y2 = yc + math.sin(math.radians(angle + 180)) * rminor
		cv2.line(myImage, (int(x1),int(y1)), (int(x2),int(y2)), (0, 170, 170), 2)

		# mark center
		xc, yc = ellipse[0]
		cv2.circle(myImage, (int(xc), int(yc)), 3, (255, 0, 0), -1)

	return myImage


# def ellipseFitMajorAxisCompare(myImage, myImageThresholded):


# def maskCellsWithNucleiOnly


def identifyCellContours(myImage, myImageThresholded, showCenterOfMass=True, perimeterColor=[0, 0, 255]):
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

	cellCount              = 0
	contours, hierarchy    = cv2.findContours(myImageThresholded.astype(np.uint8), 1, 1)
	myMaskContoursAll      = np.zeros((myImage.shape[0], myImage.shape[1]), np.uint8)
	myMaskContoursAllLabel = np.zeros((myImage.shape[0], myImage.shape[1]), np.uint8)

	for ii, cnt in enumerate(contours):
		M               = cv2.moments(cnt)
		area            = cv2.contourArea(cnt)
		perimeter       = cv2.arcLength(cnt, True)
		perimeter       = round(perimeter, 1)
		myImageContours = cv2.drawContours(myImage, [cnt], -1, perimeterColor, 1)
		
		if M['m00'] != 0.0:
			x1 = int(M['m10'] / M['m00'])
			y1 = int(M['m01'] / M['m00'])
			x1_rounded = round(x1)
			y1_rounded = round(y1)
			if showCenterOfMass==True:
				myImageContours[y1_rounded - 2 : y1_rounded + 2, x1_rounded - 2 : x1_rounded + 2] = perimeterColor
		cv2.putText(myImageContours, f'{ii + 1}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

		myMaskZeros       = np.zeros(myImage.shape, np.uint8)
		myMaskContoursTmp = cv2.drawContours(image = myMaskZeros, contours=[cnt], contourIdx = -1, color = (255, 255, 255), thickness = cv2.FILLED)
		myMaskContours                           = np.zeros((myImage.shape[0], myImage.shape[1]))
		
		myMaskContours[myMaskContoursTmp[:,:,0] == 255] = 1
		myMaskContoursAll  = myMaskContoursAll + myMaskContours

		myMaskContoursAllLabel  = myMaskContoursAllLabel + myMaskContours * (ii + 1)

	return contours, myImageContours, myMaskContoursAll, myMaskContoursAllLabel


def identifyCellContoursAndAreaOverlays(myImage, myImageThresholded, dictThresholdValues, meanFilledArea):
	# sources
	# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
	# https://de.mathworks.com/matlabcentral/answers/43435-i-couldn-t-understand-convex-area
	# https://stackoverflow.com/questions/32401806/get-mask-from-contour-with-opencv
	# https://stackoverflow.com/questions/50591442/convert-3-dim-image-array-to-2-dim
	# https://www.tutorialspoint.com/how-to-compute-the-area-and-perimeter-of-an-image-contour-using-opencv-python

	# source of main algorithm
	# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
	# User: Costas Voglis

	# https://de.mathworks.com/matlabcentral/answers/43435-i-couldn-t-understand-convex-area
	# prop.convex_area / prop.filled_area
	# convex_area (convex hull): the smallest region that satisfy two conditions: (1) it is convex (2) it contains
	# the original region.
	# => The ratio increases the less convex the area is (multiple nuclei)

	# import modules
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.measure import regionprops
	from statistics import mean

	cellCount           = 0
	contours, hierarchy = cv2.findContours(myImageThresholded.astype(np.uint8), 1, 2)
	myMaskContoursAll   = np.zeros(myImage.shape)

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
		# celladd = math.ceil(prop.filled_area)/(2291.9454545454546)
		# celladd = math.ceil((prop.filled_area)/(2276.51))
		celladd = round((prop.filled_area)/(meanFilledArea))

		if  ratio_conv_filled > dictThresholdValues["gt2CellCluster"]:
			myMaskContoursAll  = myMaskContoursAll + myMaskContours * 0.9
			cellCount=cellCount + celladd
		elif  ratio_conv_filled > dictThresholdValues["2CellCluster"]:
			myMaskContoursAll  = myMaskContoursAll + myMaskContours * 0.5
			cellCount = cellCount + celladd
		else:
			myMaskContoursAll  = myMaskContoursAll + myMaskContours * 0.2
			cellCount = cellCount + celladd
		myMaskContoursAll = cv2.drawContours(myMaskContoursAll, [cnt], -1, (0,0,255), 2)
		cv2.putText(myMaskContoursAll, f'{ii + 1}', (x1, y1 + 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (150, 150, 150), 1)
		cv2.putText(myMaskContoursAll, "%.3f" % ratio_conv_filled, (x1, y1+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 0, 0), 1)
	return cellCount, contours, myImageContours, myMaskContoursAll


def meanFilledArea(myImage, myImageThresholded, dictThresholdValues):
	# import modules
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.measure import regionprops
	from statistics import mean

	contours, hierarchy = cv2.findContours(myImageThresholded.astype(np.uint8), 1, 2)
	oneNucleiArea      	= []

	for ii, cnt in enumerate(contours):
		perimeter       = cv2.arcLength(cnt, True)
		perimeter       = round(perimeter, 1)
		myMaskZeros       = np.zeros(myImage.shape, np.uint8)
		myMaskContoursTmp = cv2.drawContours(image = myMaskZeros, contours=[cnt], contourIdx = -1, color = (255, 255, 255), thickness = cv2.FILLED)
		props = regionprops(myMaskContoursTmp, cache = False)
		prop = props[0]
		ratio_conv_filled = prop.convex_area / prop.filled_area
		if  ratio_conv_filled > dictThresholdValues["gt2CellCluster"]:
			pass
		elif  ratio_conv_filled > dictThresholdValues["2CellCluster"]:
			pass
		else:
			oneNucleiArea.append(prop.filled_area)
	meanFilledArea = mean(oneNucleiArea) 
	return meanFilledArea


def removeMinimalAreas(myImage, myImageThresholded, minimalArea):
	# sources
	# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
	# https://de.mathworks.com/matlabcentral/answers/43435-i-couldn-t-understand-convex-area
	# https://stackoverflow.com/questions/32401806/get-mask-from-contour-with-opencv
	# https://stackoverflow.com/questions/50591442/convert-3-dim-image-array-to-2-dim
	# https://www.tutorialspoint.com/how-to-compute-the-area-and-perimeter-of-an-image-contour-using-opencv-python

	# source of main algorithm
	# https://www.kaggle.com/code/voglinio/separating-nuclei-masks-using-convexity-defects
	# User: Costas Voglis

	# https://de.mathworks.com/matlabcentral/answers/43435-i-couldn-t-understand-convex-area
	# prop.convex_area / prop.filled_area
	# convex_area (convex hull): the smallest region that satisfy two conditions: (1) it is convex (2) it contains
	# the original region.
	# => The ratio increases the less convex the area is (multiple nuclei)

	# import modules
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.measure import regionprops
	from statistics import mean

	cellCount           = 0
	contours, hierarchy = cv2.findContours(myImageThresholded.astype(np.uint8), 1, 2)
	myMaskContoursAll   = np.zeros(myImage.shape)

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
		# cv2.putText(myImageContours, f'{ii + 1}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
	return contours, myImageContours, myMaskContoursAll


