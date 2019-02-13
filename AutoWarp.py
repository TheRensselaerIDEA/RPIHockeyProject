import json,pprint,cv2
import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import os
import sys
import pandas as pd
# cd \HockeyResearch\code\research code\General Warp
# These are the coordinates for the spots of the ice for an overhead view
# We will need 4 points from the picture and transform those points to its corresponding point on the overhead view
ground_truths = {'Red Line':(100,0), 'Red Line Bottom':(100,85), 
				 'Right Blue Line': (125,0), 'Right Blue Line Bottom': (125,85), 'Left Blue Line': (75,0), 'Left Blue Line Bottom': (75,85),
				 'Right Goal Line': (189,0), 'Right Goal Line Bottom': (189,85), 'Left Goal Line': (11,0), 'Left Goal Line Bottom': (11,85),
				 'Center ice':(100,42.5),
				 'Right Top Neutral Faceoff Dot':(120,20.5), 'Right Bottom Neutral Faceoff Dot':(120,64.5), 
           'Left Top Neutral Faceoff Dot':(80,20.5),	'Left Bottom Neutral Faceoff Dot':(80,64.5),
				 'Right Top Zone Faceoff Dot':(169,20.5),
			 						 'Right Bottom Zone Faceoff Dot':(169,64.5),
			 						 'Left Top Zone Faceoff Dot':(31,20.5),
			 						 'Left Bottom Zone Faceoff Dot':(31,64.5),
			 	 'Right Top Goal Post':(189,39.5),
	 						  'Right Bottom Goal Post':(189,45.5),
	 						  'Left Top Goal Post':(11,39.5),
	 						  'Left Bottom Goal Post':(11,45.5),
	 			 'Left Top Hash-Left Inner Hash Mark':(29.5,35.5),
	 			 					'Left Top Hash-Right Inner Hash Mark':(32.5,35.5),
	 			 					'Left Bottom Hash-Left Inner Hash Mark':(29.5,49.5),
	 			 					'Left Bottom Hash-Right Inner Hash Mark':(32.5,49.5),
	 			 					'Right Top Hash-Left Inner Hash Mark':(167.5,35.5),
	 			 					'Right Top Hash-Right Inner Hash Mark':(170.5,35.5),
	 			 					'Right Bottom Hash-Left Inner Hash Mark':(167.5,49.5),
	 			 					'Right Bottom Hash-Right Inner Hash Mark':(170.5,49.5)}

def distance(x, y):
    dist = math.sqrt(math.pow(y[0]-x[0], 2) + math.pow(y[1]-x[1], 2))
    return dist


# Data need to look like this
# frame.jpg: {feature1: {rink coord, real coord}, {feature2: {rink coord, real coord}}}
def init():
    with open('RushPlays.json') as f:
        data = json.load(f)
    data = pd.DataFrame(data)
    features = data['Label']
    
    # Removing frames that are not marked
    for i in range(len(data)):
        if (features[i] == 'Skip'):
            data = data[data.index != i]
    
    # Renumber the dataframe
    data = data.reset_index(drop=True)
    
    # The image names
    ids = data['External ID']
    
    rush_play = "RushPlay7/"
    #Choose which rush play you want to focus on
    for i in range(len(data)):
        if not (ids[i].startswith("RushPlay_7")):
            data = data[data.index != i]
    
    # Renumber the dataframe
    data = data.reset_index(drop=True)

    # Contains the marked locations of the image
    features = data['Label']
    # The image names
    ids = data['External ID']
    
    # Initializing the data structure that will hold the locations for each feature
    # If the feature does not exist we assign it to a negative value
    dataset = np.zeros((len(data), len(ground_truths), 2)) - 100
    
    # Assigning each feature a index for the dataset
    loc_index = {}
    j = 0
    for feature in ground_truths:
        loc_index[feature] = j
        j += 1
    
          
    for i in range(len(dataset)):
        # Getting the height of the image
        im_dir = os.path.abspath(rush_play + ids[i])
        im = cv2.imread(im_dir)
        height = len(im)
        for feature in features[i]:
            if (feature in loc_index):
                # setting the x coord
                dataset[i, loc_index[feature], 0] = features[i][feature][0]['geometry']['x']
                # setting the y coord
                dataset[i, loc_index[feature], 1] = features[i][feature][0]['geometry']['y'] 

    print(dataset)
     
    return dataset, ids, rush_play
 



        
data, ids, play = init()
# making a dictionary assigning indices of the features to names
loc_name = {}
j = 0
for feature in ground_truths:
    loc_name[j] = feature
    j += 1

loc_index = {}
j = 0
for feature in ground_truths:
    loc_index[feature] = j
    j += 1

# Showing the images and marking the locations on the image
for i in range(len(data)):
    im_dir = os.path.abspath(play + ids[i])
    im = cv2.imread(im_dir)
    test_frame = cv2.imread(im_dir)
    # Loop through the features to mark them
    for j in range(len(data[i])):
        if not (data[i,j,0] == -100 and data[i,j,1] == -100):
            coord = (int(data[i,j,0]), int(data[i,j,1]))
            cv2.circle(im, coord, 3, (0, 0, 255), -1)
            cv2.putText(im, loc_name[j], (int(data[i,j,0]) - 20, int(data[i,j,1]) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    # Must be in order of [top left, top right, bottom left, bottom right]
    # init_pts will hold the coordinates of the points on the non warped image
    init_pts = []
        
    # dest_pts will hold the coordinates of where the points are going to be transformed to
    dest_pts = []
    # Make an array of the four corners of the image to compare to
    corners = [(0,0), (0,im.shape[1]), (im.shape[0], 0), (im.shape[0], im.shape[1])]
        
    # A set to see what points we are already using for the transformation
    added_pts = set()
    #print(data[i])
    
    for corner in corners:
        k = 0
        # The current minimum point from the corner
        curr_min = ""
    
        # The and init point destination point
        curr_pt = (0, 0)
        curr_end = (0, 0)
        
        # Loop though the features
        for j in range(len(data[i])):
            point = ((int(data[i,j,0]), int(data[i,j,1])))
            if (data[i,j,0] < 0):
                continue
            if (k == 0):
                if (loc_name[j] in added_pts):
                    continue
                else:
                    min_dist = distance(point, corner)
                    curr_min = loc_name[j]
                    curr_pt = point
                    curr_end = ground_truths[loc_name[j]]
                    k += 1
                    #print(min_dist, loc_name[j])
                    continue
            dist = distance(point, corner)
            #print(dist, loc_name[j])
            if (dist < min_dist and loc_name[j] not in added_pts):
                min_dist = dist
                curr_min = loc_name[j]
                curr_pt = point
                curr_end = ground_truths[loc_name[j]]
        

        # prevents having 3 points in a line in the image        
        if (curr_min != ""):
            # for the goal posts - may have added both goal posts making it possible to have 3 points in a row
            if (curr_min == 'Right Top Goal Post'):
                added_pts.add('Right Bottom Goal Post')
            elif (curr_min == 'Right Bottom Goal Post'):
                added_pts.add('Right Top Goal Post')
            elif (curr_min == 'Left Top Goal Post'):
                added_pts.add('Left Bottom Goal Post')
            elif (curr_min == 'Left Bottom Goal Post'):
                added_pts.add('Left Top Goal Post')
            
            # The case for where we may choose both zone face off dots and a hash-mark
            # We will deal with right-side and left-side seperately
            # Right-side
            # Once we add the second point of any of the inner hash marks or the zone faceoff dots we cant add anymore
            right_hashes_zone = {'Right Top Zone Faceoff Dot', 'Right Bottom Zone Faceoff Dot',
                                 'Right Top Hash-Left Inner Hash Mark', 'Right Top Hash-Right Inner Hash Mark',
                                 'Right Bottom Hash-Left Inner Hash Mark', 'Right Bottom Hash-Right Inner Hash Mark'}
            if (curr_min in right_hashes_zone and len(added_pts.intersection(right_hashes_zone)) == 1):
                added_pts = added_pts.union(right_hashes_zone)
                
            added_pts.add(curr_min)
            init_pts.append(curr_pt)
            dest_pts.append(curr_end)
       
    # Making the lists np arrays
    init_pts = np.float32(init_pts)
    dest_pts = np.float32(dest_pts)
    
    print(init_pts)
        
    # Multiplying the destination points by 10 to form it to the frame
    dest_pts *= 5
    
    cv2.imshow('im', im)
    k = cv2.waitKey(0)
    
    try:
        xform_matrix = cv2.getPerspectiveTransform(init_pts, dest_pts)
    except:
        print("This transform did not work")
        continue
        
    persp_frame = cv2.warpPerspective(test_frame, xform_matrix, (1500, 800))
        
    cv2.imshow('Warp', persp_frame)
    k = cv2.waitKey(0)







