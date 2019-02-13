
# coding: utf-8

# # Identifying Known Rink Locations
# The objective of this program is to identify known rink locations and learn how to identify them on a sequence of images.

# In[1]:


import os 
import sys
import tensorflow as tf                                                                                        
import numpy as np  
import pandas as pd
# import pickle                                                                   
# import matplotlib as mpl
# #mpl.use('Agg')
import matplotlib.pyplot as plt   
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# #import matplotlib.axes as ax
# from sklearn.metrics import confusion_matrix    
# import itertools
# import scipy.misc
# import imageio
import json
import cv2


#download inception for its transfer layer
import inception
inception.maybe_download()

SRC_PATH = os.path.abspath("./")

from time import time 



# In[28]:


def plots(lossArrs, distArrs, valItrs, itr, save= False):
    plt.figure()
    #plt.subplot(211)
    line1 = plt.plot(range(itr), lossArrs[0], label="Train loss vals")
    # Create a legend for the first line.
    #first_legend = plt.legend( handles=[line1], loc=1 )
    #plt.subplot(212)
    plt.plot(valItrs, lossArrs[1], label="Val loss values")
    plt.title("Mean Squared Error Loss over Iterations for Validation and Training Data\n")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.legend(loc='upper right')
    if save:
        plt.savefig(graph_path + "Loss_v_Iteration")
    else:
        plt.show()
        
        
    plt.figure()
    #plt.subplot(275)
    plt.plot(range(itr), distArrs[0], label= "Train avg distance errors")
    plt.plot(valItrs, distArrs[1], label = "Val avg distance errors")
    plt.title("Average Distance Error over Iterations for Validation and Training Data\n")
    plt.xlabel("Iterations")
    plt.ylabel("Average Distance Value")
    plt.legend(loc='upper right')
    if save:
        plt.savefig(graph_path+"AvgDst_v_Iteration")
    else:
        plt.show()


#Define Variable Functions
def weight_variable(name, shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, shape=shape,
                            initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
        
#     xAxis = np.arange(0, 20.5, .5)
#     print xAxis
#     yVals = [[], [], [], [], [], [], []]
#     #print finalValPtDiff
#     #size = float(finalValPtDiff.size)
#     labels = ["head", "right shoulder", "left shoulder", "righ wrist", "left wrist", "right elbow", "left elbow"]
#     for val in xAxis:
#         for i in range(len(labels)):
#             temp = np.copy(finalValPtDiff)[:,:,i]
#             #print temp.shape
#             temp[ temp < val] = 0
#             #print temp
#             count = np.count_nonzero( temp )
#             #print count
#             yVals[i].append(1- (count/float(temp.size)))

#     plt.figure()
#     for i in range(len(labels)):
#         plt.plot(xAxis, yVals[i], label= labels[i])
#     plt.legend(loc='upper left')
#     plt.title("Accuracy Percent per Distance")
#     plt.savefig(graph_path+"Accuracy Percent per Distance")

def json_parser(json_name, key_loc_pairs, D, not_found_val =-1):
    '''
    @param json_name str of filename to open and read in a json file with rink locations 
    @param D the length of the known rink locations

    @return image_list[str] a list of all labeled image file names
    @return in_image[bool] whether each point is in the image
    '''
    with open(json_name) as f:
        data = json.load(f)

    # convert json to pandas dataframe
    data = pd.DataFrame(data)

    # remove skipped images
    data = data[data.Label != 'Skip']

    # remove images that don't have updated naming convention
    elim_lst = ["RushPlay_" in frame_name for frame_name in data["External ID"]]
    data = data[elim_lst]
    data = data.reset_index(drop=True)
    
    # N data points we can work with
    N = len(data['External ID'])

    # convert to list of image filenames
    image_list = data['External ID'].tolist()

#     print(data['Label'])
    
    # define gt_data structure and set to (not found)
    rink_locs = np.zeros((N, D, 2)) + not_found_val
    in_image = np.zeros((N,D), dtype= bool)

    
    for i in range(N):
        # Getting the height of the image
#         im_dir = os.path.abspath(rush_play + ids[i])
#         im = cv2.imread(im_dir)
#         height = len(im)
        
        '''
        for feature in data['Label'][i]:
            if (feature in loc_index):
                # setting the x coord
                rink_locs[i, loc_index[feature], 0] = features[i][feature][0]['geometry']['x']
                # setting the y coord
                rink_locs[i, loc_index[feature], 1] = features[i][feature][0]['geometry']['y']  
        '''
        for feature in key_loc_pairs.keys():
            if feature in data['Label'][i]:
                rink_locs[i, key_loc_pairs[feature], 0] = data['Label'][i][feature][0]['geometry']['x']
                # setting the y coord
                rink_locs[i, key_loc_pairs[feature], 1] = data['Label'][i][feature][0]['geometry']['y']
                in_image[i, key_loc_pairs[feature]] = 1
            '''else:
                rink_locs[i, key_loc_pairs[feature], 0] = not_found_val
                rink_locs[i, key_loc_pairs[feature], 1] = not_found_val
'''
    return image_list, in_image, rink_locs 



def statistics(rink_locs, key_loc_pairs, not_found_val= -1.):
    print("Percentage of points found:")
    
    for feature in key_loc_pairs.keys():
        rows, cols = np.where(rink_locs[:,key_loc_pairs[feature],:] != np.array([not_found_val, not_found_val], dtype = np.float))
#         print(rows)
#         print(cols)
        num = len(rows)//2 #np.count_nonzero(rink_locs[rows,key_loc_pairs['Right Goal Line'],cols]) // 2
        print("{}: {:.1f}%  ({}/{})".format(feature, 100*num/len(rink_locs), num, len(rink_locs)))

# In[2]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
JSON_DIR = os.path.abspath("./")
IMAGE_DIR = os.path.join(SRC_PATH, "./images")
stime = time()


# ### Load the json labels
# The json file contains all labeled rink locations, and labels not in the images should be labeled as (-1,-1)

# ### Declare list of labels to predict
# Out of the possible known rink locations, these are the ones that appear in our dataset

# In[3]:


known_rink_locs = ['Red Line', 
                    'Red Line Bottom',
                'Right Blue Line', 
                    'Right Blue Line Bottom', 
                    'Left Blue Line', 
                    'Left Blue Line Bottom',
                'Right Goal Line', 
                    'Right Goal Line Bottom', 
                    'Left Goal Line',
                    'Left Goal Line Bottom',
                'Center ice',
                'Right Top Neutral Faceoff Dot',
                    'Right Bottom Neutral Faceoff Dot',
                    'Left Top Neutral Faceoff Dot',
                    'Left Bottom Neutral Faceoff Dot',
                'Right Top Zone Faceoff Dot',
                    'Right Bottom Zone Faceoff Dot',
                    'Left Top Zone Faceoff Dot',
                    'Left Bottom Zone Faceoff Dot',
                'Right Top Goal Post',
                    'Right Bottom Goal Post',
                    'Left Top Goal Post',
                    'Left Bottom Goal Post',
                'Left Top Hash-Left Inner Hash Mark',
                    'Left Top Hash-Right Inner Hash Mark',
                    'Left Bottom Hash-Left Inner Hash Mark',
                    'Left Bottom Hash-Right Inner Hash Mark',
                    'Right Top Hash-Left Inner Hash Mark',
                    'Right Top Hash-Right Inner Hash Mark',
                    'Right Bottom Hash-Left Inner Hash Mark',
                    'Right Bottom Hash-Right Inner Hash Mark']
print(len(known_rink_locs))
known_rink_loc_set = set(known_rink_locs)

D = len(known_rink_locs)
key_loc_pairs = {'Red Line': 0, 
                    'Red Line Bottom':1,
                'Right Blue Line': 2, 
                    'Right Blue Line Bottom': 3, 
                    'Left Blue Line': 4, 
                    'Left Blue Line Bottom': 5,
                'Right Goal Line': 6, 
                    'Right Goal Line Bottom': 7, 
                    'Left Goal Line': 8,
                    'Left Goal Line Bottom': 9,
                'Center ice': 10,
                'Right Top Neutral Faceoff Dot': 11,
                    'Right Bottom Neutral Faceoff Dot': 12,
                    'Left Top Neutral Faceoff Dot':13,
                    'Left Bottom Neutral Faceoff Dot':14,
                'Right Top Zone Faceoff Dot': 15,
                    'Right Bottom Zone Faceoff Dot':16,
                    'Left Top Zone Faceoff Dot': 17,
                    'Left Bottom Zone Faceoff Dot': 18,
                'Right Top Goal Post': 19,
                    'Right Bottom Goal Post': 20,
                    'Left Top Goal Post': 21,
                    'Left Bottom Goal Post': 22,
                'Left Top Hash-Left Inner Hash Mark': 23,
                    'Left Top Hash-Right Inner Hash Mark': 24,
                    'Left Bottom Hash-Left Inner Hash Mark': 25,
                    'Left Bottom Hash-Right Inner Hash Mark': 26,
                    'Right Top Hash-Left Inner Hash Mark': 27,
                    'Right Top Hash-Right Inner Hash Mark': 28,
                    'Right Bottom Hash-Left Inner Hash Mark': 29,
                    'Right Bottom Hash-Right Inner Hash Mark': 30}
print(len(key_loc_pairs.keys()))
assert (len(key_loc_pairs.keys()) == len(known_rink_locs))
for key in key_loc_pairs.keys():
    loc = key_loc_pairs[key]
    assert known_rink_locs[loc] == key


# ## Inception
# We will use the second last layer of inception as the input to the LSTM network.  This way, we can "understand" the input images, without the need to train on LARGE amounts of data

# In[4]:


#use this model to get the transfer layer for detecting image similarities
model = inception.Inception()

#use this command to get the transfer layer values of the image
# model_TL.transfer_values(image= gt_im)


# In[ ]:





# ### Weights
# Define weights to output layer from transfer layer

# In[5]:





# ### Data split into test and train
# Separate data for preparation to training

# In[8]:


# print "Splitting data into Training and Validation Sets"
# sDTrainValSplitTime = time()
# testSet = True
# valPercent = .2  #.11
# seed = 42
# xTrain, xVal, yTrain, yVal = train_test_split(xData, yData, test_size= valPercent, random_state= seed, shuffle= True)
# if testSet:
#     testPercent = .3 #taken from valset


# ### JSON Parsing
# Preepare the y-data labels from json file.  The xy labels in the image should be defined as their ground truth locations, while xy locations not in the frame should be defined as a variable $v*(-1,-1)$

# In[22]:





# In[10]:





# In[23]:

NOT_FOUND_VAL = -1
image_list, y_in_image, rink_locs = json_parser(os.path.join(SRC_PATH, 
                                'RushPlays.json'), key_loc_pairs, D, not_found_val= NOT_FOUND_VAL)

statistics(rink_locs, key_loc_pairs, NOT_FOUND_VAL)


# In[12]:


# read in known locations

# print(data['Label'])
pts = rink_locs[0]
for feature in key_loc_pairs.keys():
    loc = key_loc_pairs[feature]
    x,y = pts[loc]
    if x != -1 and y != -1:
        plt.scatter(x,y, label= feature)
#     print( value )
plt.imshow(mpimg.imread( os.path.join(IMAGE_DIR, image_list[0]) ))



images = [mpimg.imread(os.path.join(IMAGE_DIR, im_name)) for im_name in image_list] #[mpimg.imread(os.path.join(IMAGE_DIR, im_name)) for im_name in image_names]



# In[13]:


for feature in key_loc_pairs.keys():
    loc = key_loc_pairs[feature]
    x,y = pts[loc]
    if x != -1 and y != -1:
        plt.scatter(x,y, label= feature)
plt.legend()
plt.imshow(mpimg.imread( os.path.join(IMAGE_DIR, image_list[0]) ))


# In[34]:


#Create Tensorflow Datasets for Training and Validation
batchSize = 5
# seq_len = 30
outputs = (len(known_rink_locs), 2)

IMAGE_DIR = os.path.abspath("./RushPlay2/")
# print(os.listdir(IMAGE_DIR))
# images = [mpimg.imread(os.path.join(IMAGE_DIR, im)) for im in data['External ID'][:3]]

# transfer_values = inception.process_images(inception.transfer_values, images= images)
# image_data = np.array([model.transfer_values(image= image_) for image_ in images])
# transfer_values = model.process_images(transfer_values_cache(),  )
# transfer_values = model_TL.transfer_values(images= gt_im)
# tv_size = image_data.shape[-1]
tv_size = 2048
loc_len = len(known_rink_locs)

#Define variables
# x = tf.placeholder(shape=[None, seq_len, 64, 64, 3], dtype=tf.float32) 
x = tf.placeholder(shape= [None, tv_size], dtype= tf.float32)
y = tf.placeholder(shape=[None, loc_len, 2], dtype=tf.int32) 
y_in_img = tf.placeholder(shape= [None, loc_len], dtype= tf.float32)
y_float = tf.cast(y, dtype= tf.float32)


# In[27]:


X_train_ims, X_test_ims, Y_in_image_train, Y_in_image_test, Y_train, Y_test =  train_test_split(images, y_in_image, rink_locs, test_size=0.2, random_state=42)
X_train = np.array([model.transfer_values(image= image_) for image_ in X_train_ims])
X_test = np.array([model.transfer_values(image= image_) for image_ in X_test_ims])
print(X_train.shape, X_test.shape)





    
    
    
    
    
    


# In[29]:


W_out = weight_variable("output_weights", [tv_size, loc_len*2])
W_in_img = weight_variable("output_in_img_weights", [tv_size, loc_len])



# In[18]:


W_star = tf.identity(W_out)


# In[35]:


y_hat = tf.matmul(x, W_out)
y_in_im_hat = tf.sigmoid(tf.matmul(x, W_in_img))
in_im_hat = y_in_im_hat > .5

# tf.cond(y_in_im_hat > .5, 1, 0)

y_hat = tf.reshape(y_hat, [-1, y_hat.shape[1]//2, 2])
print(y_hat)
print(y_in_im_hat)


# In[38]:


# loss function
eta = 1e-6
mse_loss = tf.losses.mean_squared_error(y_float*tf.reshape(y_in_img, [-1,loc_len, 1]), y_hat*tf.reshape(y_in_img, [-1,loc_len, 1]))
mse_in_img_loss = tf.losses.mean_squared_error(y_in_img, y_in_im_hat)

alpha = 1.0
train_step = tf.train.AdamOptimizer(eta).minimize(mse_loss + alpha*mse_in_img_loss)
pointDiff = tf.norm(y_hat - y_float, axis= 2)
totDiff = tf.reduce_mean(tf.reduce_mean(pointDiff, axis= 0))

predict_op = tf.cast(y_hat, dtype= tf.int64)
pred_in_image = tf.cast(in_im_hat, dtype= tf.bool)
train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_loss)
print(pointDiff.shape)
print(totDiff.shape)
print(mse_loss.shape)

##########  Declare Variables for Saving ############################## 
# Save the model
tf.get_collection('validation_nodes')

# Add opts to the collection
tf.add_to_collection('validation_nodes', x)
tf.add_to_collection('validation_nodes', y)
tf.add_to_collection('validation_nodes', predict_op)
##########  End Variable Declaration for Saving ############################## 


# In[40]:


#########################################################
##  The Model                                          ##
#########################################################

# start training
saver = tf.train.Saver()
# Run one pass over the validation dataset.
#session.run(train_itr.initializer)
lossArrs = [[], [], []] #train, val, test
distArrs = [[], [], []] #train, val, test

minDist = [np.inf, np.inf, np.inf] #train, val, test
minLoss = [np.inf, np.inf, np.inf] #train, val, test


variableValEval = False #False for faster runtime
valEvalInterval = 300
valItrs = []
# valBatchPercent = .2
# valBatchSize = 20


sRT = time()
itr = 0
logInterval = 300
epochs = 1 #2000
max_iterations = None #101 #200000
# keepRate = .25
accLvl = .9
batchSize = 5
save_path = "./my_model/my_model"


graph_path = "./graphs/"
#feedDictItrTrain = {xTrainPH: xTrain, yTrainPH: yTrain}

# Launch the graph in a session.
session = tf.Session()


# In[42]:


session.run(tf.global_variables_initializer())
#train_handle = session.run(train_itr.string_handle())
#val_handle = session.run(val_itr.string_handle())
print
for i in range(epochs):
    if max_iterations != None and max_iterations <= itr:
        print("here", itr)
        break
    #if i > 0:
    #shuffle and create new batches
    X_train, X_train_ims, Y_train, Y_in_image_train = shuffle(X_train, X_train_ims, Y_train, Y_in_image_train, random_state= i)
    
    
#     X_train_batch = np.array_split(X_train, len(X_train)/batchSize)
#     Y_train_batch = np.array_split(Y_train, len(Y_train)/batchSize)

    
    
    '''
    session.run(train_itr.initializer, feed_dict= feedDictItrTrain)
    j = 0
    while True:
        #print j
        try:
            batch = session.run(next_elementTrain, feed_dict={handle: train_handle})
            #print batch[0].shape
            #print batch[1].shape
        except tf.errors.OutOfRangeError:
              break
    '''
    for j in range(len(np.ceil(X_train/batchSize))-1):
#         if max_iterations != None and max_iterations <= itr:
#             break  
        #print batch[1]
        
                #feedDict = {x: batch[0], y: batch[1], keep_prob: 0.5}
        #feedDict = {x: batch[0], y: batch[1], keep_prob: keepRate}
        if j+batchSize > len(X_train):
            X_train_batch = X_train[j:]
            Y_train_batch = Y_train[j:]
            Y_in_train_batch = Y_in_image_train[j:]
        elif j == len(X_train):
            break
        else:
            X_train_batch = X_train[j:j+batchSize]
            Y_train_batch = Y_train[j:j+batchSize]
            Y_in_train_batch = Y_in_image_train[j:j+batchSize]
        
        
        feedDict = {x: X_train_batch, y: Y_train_batch, y_in_img: Y_in_train_batch}
        trainLoss, trainDist, _ = session.run([mse_loss, totDiff, train_step], feed_dict= feedDict)
        lossArrs[0].append(trainLoss)
        distArrs[0].append(trainDist)
        if trainDist < minDist[0]:
            minDist[0] = trainDist
        if trainLoss < minLoss[0]:
            minLoss[0] = trainLoss
#         if itr%trainAccInterval == 0:
#             trainItrs.append(itr)
#             accArrs[0].append(accuracy.eval(feed_dict= feedDict))
#             if maxAcc[0] < accArrs[0][-1]:
#                 maxAcc[0] = accArrs[0][-1]
        if (itr+1)%valEvalInterval == 0:
            valItrs.append(itr)
            
#             xValBatch, _, yValBatch, _ = train_test_split(X_test, Y_test, test_size= 1-valBatchPercent, shuffle= True)
            feedDictVal = { x: X_test, y: Y_test, y_in_img: Y_in_image_test}
            
            valLoss, valDiff, valPointDiff = session.run([mse_loss, totDiff, pointDiff], feed_dict= feedDictVal)
            distArrs[1].append(valDiff)
            lossArrs[1].append(valLoss)
            if valDiff < minDist[1]:
                minDist[1] = valDiff
                #test accuracy is a new maximum.  Save the graph
                # this saver.save() should be within the same tf.Session() after the training is done
                #Only save if greater than a threshold... that way we don't waste time
#                 if variableValEval:
#                     if minAcc[1] > accLvl:
#                         saver.save(session, save_path)
#                         valEvalInterval = max(1, round(np.sqrt(valEvalInterval)))
#                     elif minAcc[1] < .8*accLvl:
#                         valEvalInterval = max(50, round(np.sqrt(valEvalInterval)))
#                     else:
#                         valEvalInterval = max(5, round(np.sqrt(valEvalInterval)))
#                 else:
#                     saver.save(session, save_path)
            if valLoss < minLoss[1]:
                minLoss[1] = valLoss
            
                # save the weights
                W_star_ = session.run(W_star)
                
        if (itr+1)%logInterval == 0:
            #dataset = dataset.shuffle(buffer_size=10000)
            #feedDict= { x:batch[0], y: batch[1], keep_prob: 1.0}
            
            feedDict= { x: X_train_batch, y: Y_train_batch, y_in_img: Y_in_train_batch}
            trainLoss, trainDiff, trainPtDiff = session.run([mse_loss, totDiff, pointDiff], feed_dict= feedDict)
            if (itr+1)%valEvalInterval != 0:
                #session.run(val_itr.initializer)
                #dVal = session.run(next_elementVal, feed_dict= {handle: val_handle})
                #feedDictVal = { x: dVal[:len(dVal)/2][0], y: dVal[len(dVal)/2:][0], keep_prob: 1.0}
#                 xValBatch, _, yValBatch, _ = train_test_split(xVal, yVal, test_size= 1-valBatchPercent, shuffle= True)
                feedDictVal = { x: X_test, y: Y_test}
                valLoss, valDist, valPointDist = session.run( [mse_loss, totDiff, pointDiff], feedDictVal)
            print("Epoch %d, step %d,     training dist: %g"%(i, j, trainDiff))
            print("                            val dist: %g"%(valDiff))
            print("                      training loss: %g"%(trainLoss))
            print("                     validation loss: %g"%(valLoss))
            print("                  min validation dist: %g"%(minDist[1]))
            print("                       min train dist: %g"%(minDist[0]))
#             plots(lossArrs, distArrs, valItrs, itr+1, save= False)
            print("                                 RT = %.2f"%(time()-sRT))
            sys.stdout.flush()
        itr +=1   
plots(lossArrs, distArrs, valItrs, itr, save= False)



# In[55]:


test_preds, testDiff, test_inframe = session.run([predict_op, totDiff, in_im_hat], feed_dict= {x: X_test, y: Y_test, y_in_img: Y_in_image_test})
# print(test_inframe)
def print_n_classifs(n, ims, labels, preds, in_im_pred):
    locs = np.random.choice(range(len(ims)), n, replace= False)
    
    for i in locs:
       
        f = plt.figure(figsize=(15,15))
        #f.subplot(121)
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        pts = preds[i]
        for feature in key_loc_pairs.keys():
            loc = key_loc_pairs[feature]
#             if (np.array(loc) <= 0).any():
#                 continue
            if in_im_pred[i][loc] == False:
                continue
            x,y = pts[loc]
            if x < 0 or y < 0:
                continue
            ax.scatter(x,y, label= feature)
            ax2.scatter(x,y, label= feature)
        ax.imshow(ims[i])
        ax2.imshow(ims[i])
        ax2.legend()
        
        
        
        plt.figure()
        pts = labels[i]
        for feature in key_loc_pairs.keys():
            loc = key_loc_pairs[feature]
            x,y = pts[loc]
            if x > 0 or y > 0:
                plt.scatter(x,y, label= feature)
        plt.imshow(ims[i])
        plt.legend()
        plt.show()
        
n = 1
np.random.seed(42)
print_n_classifs(n, X_test_ims, Y_test, test_preds, test_inframe)


# In[57]:


num = 5
train_preds, trainDiff, train_in_im_preds = session.run([predict_op, totDiff, in_im_hat], feed_dict= {x: X_train, y: Y_train})
print_n_classifs(n, X_train_ims, Y_train, train_preds, train_in_im_preds)

