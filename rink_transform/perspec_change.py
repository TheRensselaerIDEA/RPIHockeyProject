import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from pylab import rcParams
import os

# set the figsize and dpi globally for all images
#rcParams['figure.figsize'] = (16, 16)
rcParams['figure.figsize'] = (20, 8.5)
rcParams['figure.dpi'] = 300


IM_DIREC = os.path.abspath("./")

# read test frame
# cvtColor_flip converts cv2 BGR image to numpy RGB image
test_frame = cv2.cvtColor(cv2.imread(os.path.join(IM_DIREC, "camera_view_w_box.png")),cv2.COLOR_BGR2RGB)
#test_frame = cv2.cvtColor(cv2.imread('C:/Users/tmorgan/Desktop/Data Analytics/Sports/Hockey/frame907.jpg'),cv2.COLOR_BGR2RGB)
#plt.imshow(test_frame)
#plt.show()

# Make before and after corner picks. Coordinate order is horizontal, vertical.
# original is pixel locations from the image frame
# for test_frame
#orig_pts = np.float32([[423.0, 123.0], [1018.0, 128.0], [275.0, 436.0], [1097.0, 432.0]])
orig_pts = np.float32([[423.0, 123.0], [1018.0, 128.0], [0.0, 990.0], [1250.0, 960.0]])

# for frame907
#orig_pts = np.float32([[298.0, 127.0], [1175.0, 148.0], [612.0, 352.0], [1230, 360.0]])
#print(test_frame[305,130])

# destination is rink pixels. Rink is 200 feet horizontal by 85 feet vertical, pixels are 2000 horizontal by 850 vertical
# for test_frame
#dest_pts = np.float32([[1000, 0], [1250, 0], [1000, 425], [1250, 425]])
UL_dest = (1000,0)
UR_dest = (1250,0)
#LL_dest = (1000,425)
#LR_dest = (1250,425)
LL_dest = (1000,845)
LR_dest = (1250,845)

# for frame907
#UL_dest = (750,0)
#UR_dest = (1250,0)
#LL_dest = (1000,425)
#LR_dest = (1250,425)
dest_pts = np.float32([UL_dest, UR_dest, LL_dest, LR_dest])


# verify corner picks
test_frame_lines = test_frame.copy()
cv2.line(test_frame_lines, tuple(orig_pts[0]), tuple(orig_pts[1]), (255,0,0), 2)
cv2.line(test_frame_lines, tuple(orig_pts[1]), tuple(orig_pts[3]), (255,0,0), 2)
cv2.line(test_frame_lines, tuple(orig_pts[3]), tuple(orig_pts[2]), (255,0,0), 2)
cv2.line(test_frame_lines, tuple(orig_pts[2]), tuple(orig_pts[0]), (255,0,0), 2)

plt.imshow(test_frame_lines)
plt.show()

# Get perspective transform
xform_matrix = cv2.getPerspectiveTransform(orig_pts, dest_pts)
print(xform_matrix)

# warp image with xform_matrix
persp_frame = cv2.warpPerspective(test_frame, xform_matrix, (2000, 850), borderValue=(242, 253, 255))

# show the transformed image
#plt.imshow(persp_frame)
#plt.show()

# add rink outline, key on ice makers, and grid lines to the image space
# start by copying the warped image to a new image array
persp_frame_lines = persp_frame.copy()
#plt.imshow(persp_frame_lines)
#plt.show()

# yellow rink border at base of the boards
cv2.line(persp_frame_lines, (100, 0), (1900, 0), (242,250,10), 15)
cv2.line(persp_frame_lines, (0, 100), (0, 750), (242,250,10), 15)
cv2.line(persp_frame_lines, (100, 850), (1900, 850), (242,250,10), 15)
cv2.line(persp_frame_lines, (2000, 100), (2000, 750), (242,250,10), 15)

# Goal lines
cv2.line(persp_frame_lines, (110, 0), (110, 850), (255,0,0), 4)
cv2.line(persp_frame_lines, (1890, 0), (1890, 850), (255,0,0), 4)

# Red line
cv2.line(persp_frame_lines, (1000, 0), (1000, 850), (255,0,0), 4)

# Blue Lines
cv2.line(persp_frame_lines, (750, 0), (750, 850), (0,0,255), 4)
cv2.line(persp_frame_lines, (1250, 0), (1250, 850), (0,0,255), 4)

# Vertical grid lines every 10 feet 
for loc in range(100, 2000, 100):
	cv2.line(persp_frame_lines, (loc, 0), (loc, 850), (0,0,0), 2)
'''
cv2.line(persp_frame_lines, (200, 0), (200, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (300, 0), (300, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (400, 0), (400, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (500, 0), (500, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (600, 0), (600, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (700, 0), (700, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (800, 0), (800, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (900, 0), (900, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1100, 0), (1100, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1200, 0), (1200, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1300, 0), (1300, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1400, 0), (1400, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1500, 0), (1500, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1600, 0), (1600, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1700, 0), (1700, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1800, 0), (1800, 850), (0,0,0), 2)
cv2.line(persp_frame_lines, (1900, 0), (1900, 850), (0,0,0), 2)
'''

# Horizontal lines every 10 feet from the top of the image down
for loc in range(100, 900, 100):
	cv2.line(persp_frame_lines, (0, loc), (2000, loc), (0,0,0), 2)
'''
cv2.line(persp_frame_lines, (0, 200), (2000, 200), (0,0,0), 2)
cv2.line(persp_frame_lines, (0, 300), (2000, 300), (0,0,0), 2)
cv2.line(persp_frame_lines, (0, 400), (2000, 400), (0,0,0), 2)
cv2.line(persp_frame_lines, (0, 500), (2000, 500), (0,0,0), 2)
cv2.line(persp_frame_lines, (0, 600), (2000, 600), (0,0,0), 2)
cv2.line(persp_frame_lines, (0, 700), (2000, 700), (0,0,0), 2)
cv2.line(persp_frame_lines, (0, 800), (2000, 800), (0,0,0), 2)
'''

height = 100
width = 100
# Ellipse parameters
radius = 50
center = (width / 2, height - 25)
axes = (radius, radius)
angle = 0
startAngle = 0
endAngle = 90
thickness = 15
#cv2.ellipse(persp_frame_lines, center=center, axes=(radius,radius), angle=0, startAngle=0, endAngle=90, color=(242,250,10), thickness=15)

#pfl_bd = cv2.copyMakeBorder(persp_frame_lines,10,10,10,10,cv2.BORDER_CONSTANT,value = (242,250,10))

plt.imshow(persp_frame_lines)
plt.show()

#combined_image = np.hstack((brien_drawing_smaller, drawing))
