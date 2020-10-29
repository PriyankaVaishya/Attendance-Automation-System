import cv2
import numpy as np
#import image
def contour(image):
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# cv2.imshow('gray',gray)
	# cv2.waitKey(0)

	#binary
	ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
	# cv2.imshow('second',thresh)
	# cv2.waitKey(0)

	#dilation
	kernel = np.ones((5,100), np.uint8)

	img_dilation = cv2.dilate(thresh, kernel, iterations=1)
	# img_dilation = cv2.erode(img_dilation, kernel, iterations=1)
	cv2.imshow('dilated',img_dilation)
	cv2.waitKey(0)

	#find contours
	im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)

		# Getting ROI
		roi = image[y:y+h, x:x+w]
		area=w*h
		# show ROI
		# cv2.imshow('segment no:'+str(i),roi)
		if h<w and h> 15 and 100 < w < 500 and area < 12000 :
			
			cv2.rectangle(image,(x,y-3),( x + w, y + h+3 ),(0,255,0),1)
			cv2.resize(roi,(100,100))
			grayy= cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
			ret,threshh = cv2.threshold(grayy,127,255,cv2.THRESH_BINARY_INV)
			# kernel = np.ones((5,5), np.uint8)

			# threshh= cv2.dilate(threshh, kernel, iterations=1)
			# cv2.imshow('inner',threshh)
			# cv2.waitKey(0)
			im22,ctrss, hierr = cv2.findContours(threshh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			sorted_ctrss = sorted(ctrss, key=lambda ctr: cv2.boundingRect(ctr)[0])
			for ii, ctrr in enumerate(sorted_ctrss):
				# Get bounding box
				xx, yy, ww, hh = cv2.boundingRect(ctrr)

				# Getting ROI
				roii = roi[yy:yy+hh, xx:xx+ww]
				# show ROI
				# cv2.imshow('segment no:'+str(i),roi)
				if  1 < ww and 4 < hh :
					cv2.rectangle(roi,(xx-1,yy-2),( xx + ww+1, yy + hh+2 ),(0,255,0),1)

			cv2.imshow('marked ',roi)
			cv2.waitKey(0)
	cv2.imshow('marked areas',image)
	cv2.waitKey(0)
	cv2.imwrite('finall.jpg',image)
	return image


	#sort contours
