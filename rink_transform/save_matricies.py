# import the necessary packages
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from pylab import rcParams
import argparse
import os
import csv
import pandas as pd

# set the figsize and dpi globally for all images
rcParams['figure.figsize'] = (20, 8.5)
#rcParams['figure.figsize'] = (10, 4.25)
rcParams['figure.dpi'] = 300

players = {}
files = next(os.walk("RushPlay3"))[2] #reads through the file directory of images
print(files)
N = len(files) #sets the number of images in the directory to N
N = N * 2 #since the frames are every other, there are half as many files as frames, so to make up for this we multiple the number of files by 2 to equal the amount of frames
l = 1
global z
z = 0
#while the frame number is less than the total number of frames
while (l < N and z==0):
	# initialize the list of reference points and boolean indicating
	# whether corrdinate grab is being performed or not	
	refPt = []
	PlayerPt = []
	CamRef = []
	RinkRef = []
	cropping = False
	global i
	i = 0
	global j
	j = 0
	global x
	x = 1
	def click_and_grab(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, cropping, i
	
		rcParams['figure.figsize'] = (10, 4.25)
		rcParams['figure.dpi'] = 300
		
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			print(i)
			#print(refPt[i][0] * 2)
			#print(refPt[i][1] * 2)
			#refPt[i][0] = x
			#refPt[i][1] = y
			#refPt = [(x, y)]
			refPt.append((x, y))
			print('ref pt %d, x, y'%i)
			print(refPt[i][0] * 2)
			print(refPt[i][1] * 2)
			cv2.circle(image, center=refPt[i], radius=5, color=(0,255,0), thickness=-2)
			text = "RefPt %d"%i
			cv2.putText(image, text, (refPt[i][0],refPt[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), lineType=cv2.LINE_AA) 
			cropping = True
			i += 1
	 
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# replot image
			cv2.imshow("image", image)
	
	def player(event, x, y, flags, param):
		# grab references to the global variables
		global PlayerPt, cropping, j
	
		rcParams['figure.figsize'] = (10, 4.25)
		rcParams['figure.dpi'] = 300
		
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			print(j)
			#print(refPt[i][0] * 2)
			#print(refPt[i][1] * 2)
			#refPt[i][0] = x
			#refPt[i][1] = y
			#refPt = [(x, y)]
			PlayerPt.append((x, y))
			print('Player %d, x, y'%j)
			print(PlayerPt[j][0] * 2)
			print(PlayerPt[j][1] * 2)
			cv2.circle(image, center=PlayerPt[j], radius=5, color=(0,255,0), thickness=-2)
			text = "Player %d"%j
			cv2.putText(image, text, (PlayerPt[j][0],PlayerPt[j][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), lineType=cv2.LINE_AA) 
			cropping = True
			j += 1
	 
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# replot image
			cv2.imshow("image", image)
	
	# construct the argument parser and parse the arguments
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--image", required=True, help="Path to the image")
	#ap.add_argument("-i", "--image", "--image2",required=True, help="Path to the image")
	#args = vars(ap.parse_args())
	 
	# load the camera image
	#src = cv2.imread(args["image"])
	src = cv2.imread('RushPlay3/' + files[l])

	# make half size version to fit screen better
	image = cv2.resize(src, (0,0), fx=0.5, fy=0.5)
	#image = src
	
	# clone a copy of the half size image 
	clone = image.copy()
	
	# Set-up mouse event 
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_grab)
	
	print("camera view points")
	 
	# Select camera image reference points in the order: UL, UR, LL, LR 
	# keep looping until the 'q' key is pressed
	while (True and x==1):
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the selected points
		#if key == ord("r"):
			#image = clone.copy()
	 
		# if the 'q' key is pressed, break from the loop
		# if the 's' key is pressed, stop the program
		# if the 'u' key is pressed, undoes the frames from the beginning		
		#elif key == ord("q"):
		if key == ord("q"):
			x=2
			break
		elif key == ord("s"):
			print ("Player Coordinates: \n",players)
			z=1
			break
		elif key == ord("u"):
			x = 1
			break

	if z==1:
		cv2.destroyAllWindows()
		break
	# Reference point selection done, close image window
	cv2.destroyAllWindows()
	
	if x==2:
		# Repeat for the rink template
		# load the rink template image
		dst = cv2.cvtColor(cv2.imread('Overhead_Template.jpeg'),cv2.COLOR_BGR2RGB)
		
		# make half size version to fit screen better
		image = cv2.resize(dst, (0,0), fx=0.5, fy=0.5)
		#image = dst
		
		# clone a copy of the half size image 
		clone = image.copy()
		
		# Set-up mouse event 
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", click_and_grab)
		 
		print("overhead view points")
		
		# Select rink template image reference points in the order: UL, UR, LL, LR 
		# keep looping until the 'q' key is pressed
	while (True and x==2):
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the selected points
		#if key == ord("r"):
			#image = clone.copy()
	 
		# if the 'q' key is pressed, break from the loop
		# if the 's' key is pressed, stop the program
		# if the 'u' key is pressed, undoes the frames from the beginning		
		#elif key == ord("q"):
		if key == ord("q"):
			x=3
			break
		elif key == ord("s"):
			print ("Player Coordinates: \n",players)
			z=1
			break
		elif key == ord("u"):
			x=2
			break
	if z==1:
		cv2.destroyAllWindows()
		break			
	# Reference point selection done, close image window
	cv2.destroyAllWindows()
	
	if x==3:
		# Set reference points for mapping transform calculation
		# Since the coordinates were grabbed from the half size images, double all
		# values before generating the mapping matrix and applying to the full size images
		UL_src = (refPt[0][0] * 2, refPt[0][1] * 2)
		UR_src = (refPt[1][0] * 2, refPt[1][1] * 2)
		LL_src = (refPt[2][0] * 2, refPt[2][1] * 2)
		LR_src = (refPt[3][0] * 2, refPt[3][1] * 2)
		
		UL_dst = (refPt[4][0] * 2, refPt[4][1] * 2)
		UR_dst = (refPt[5][0] * 2, refPt[5][1] * 2)
		LL_dst = (refPt[6][0] * 2, refPt[6][1] * 2)
		LR_dst = (refPt[7][0] * 2, refPt[7][1] * 2)
		
		print("Camera View Reference Points")
		print('UL_src')
		print(UL_src)
		print('UR_src')
		print(UR_src)
		print('LL_src')
		print(LL_src)
		print('LR_src')
		print(LR_src)
		
		print("Overhead Rink View Equivalent Landmark Points")
		print('UL_dst')
		print(UL_dst)
		print('UR_dst')
		print(UR_dst)
		print('LL_dst')
		print(LL_dst)
		print('LR_dst')
		print(LR_dst)
		
		dest_pts = np.float32([UL_dst, UR_dst, LL_dst, LR_dst])
		orig_pts = np.float32([UL_src, UR_src, LL_src, LR_src])
		
		# Calculate mapping transform
		H = cv2.getPerspectiveTransform(orig_pts, dest_pts)
		print('Mapping array')
		print(H)
		
		# warp the full size camera view image with the mapping tranform
		warped_src = cv2.warpPerspective(src, H, (2000, 850), borderValue=(242, 253, 255))
		
		
		#warped_src_half_size = cv2.resize(warped_src, (0,0), fx=0.5, fy=0.5) 
		#plt.title("Warped Camera View", fontsize=10)
		#plt.imshow(warped_src)
		#plt.imshow(warped_src_half_size)
		#plt.show()
		
		# Overlay the rink overhead view onto the warped camera view
		overhead = cv2.addWeighted(dst,0.2,warped_src,0.5,0)
		
		# Plot combined views
		#plt.title("Camera View Overlain on Template", fontsize=10)
		#plt.imshow(overhead)
		#plt.show()
		
		# make half size version to fit screen better
		image = cv2.resize(overhead, (0,0), fx=0.5, fy=0.5)
		
		# clone a copy of the half size image 
		clone = image.copy()
		
		# Set-up mouse event 
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", player)
		 
		# Select player location points on the overhead view
		print("Player Locations") 
	# keep looping until the 'q' key is pressed
	while (True and x==3):
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the selected points
		#if key == ord("r"):
			#image = clone.copy()
	 
		# if the 'q' key is pressed, break from the loop
		# if the 's' key is pressed, stop the program
		# if the 'u' key is pressed, undoes the frames from the beginning
		#elif key == ord("q"):
		if key == ord("q"):
			x=4
			break
		elif key == ord("s"):
			z=1
			print ("Player Coordinates: \n",players)
			break
		elif key == ord("u"):
			x=3
			break

	if z==1:
		cv2.destroyAllWindows()
		break
	# Reference point selection done, close image window
	cv2.destroyAllWindows()
	
	if x==4:
		#multiplied by two because the image size is cut in half when displaying
		players['RushPlay_3_%0.5d'%l] = PlayerPt * 2
		# goes through every 10 frames
		l += 10

#puts the dictionary into a CSV file
pd.DataFrame(players).T.reset_index().to_csv('RushPlay3.csv', header=False, index=False)