#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import libraries
import tensorflow as tf
import numpy as np
import cv2
import copy
import PIL
import pickle
import time 
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import os
###################
################
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model


# In[4]:


#load FaceNet model
FaceNet=tf.keras.models.load_model('facenet_keras.h5',compile=False)

#load trained YOLO model
config_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny_2000.weights"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#specifiy the image size
image_size = 160


# In[5]:


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(img,margin):
    aligned_images = []
    x,y,h,w=0,0,0,0
    
    #keep the original height and width, Caffe model require resizing to 300*300
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255, (160, 160), swapRB=True, crop=False)

    #pass the image to the model
    net.setInput(blob)

    #extract
    faces = net.forward(ln)

    #empty the buffers
    boxes, confidences, class_ids,names = [], [], [], []
    
    # loop over each of the layer outputs
    for output in faces:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            confidence = detection[5]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                x,y,width,height=abs(x),abs(y),abs(width),abs(height)
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                
    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    
    if(len(idxs)>0):
        #empty the buffers
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            face_cropped=img[y:y+h,x:x+w]
            aligned = resize(face_cropped, (image_size, image_size), mode='reflect')
            aligned_images.append(aligned)
    return np.array(aligned_images), (x, y, w, h)

def calc_embs(img, margin=10, batch_size=1):
    dummy, box=load_and_align_images(img, margin)
    if(dummy.shape[0]!=0):
        aligned_images = prewhiten(dummy)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(FaceNet.predict_on_batch(aligned_images[start:start+batch_size]))
        embs = l2_normalize(np.concatenate(pd))
    else:
        embs=np.zeros((128,1))
    return embs, box

def calc_dist(img0, img1):
    return distance.euclidean(img0, img1)

# def calc_dist_plot(img_name0, img_name1):
#     print(calc_dist(img_name0, img_name1))
#     plt.subplot(1, 2, 1)
#     plt.imshow(imread(data[img_name0]['image_filepath']))
#     plt.subplot(1, 2, 2)
#     plt.imshow(imread(data[img_name1]['image_filepath']))


# In[6]:


#pre definied varaiables
CONFIDENCE = 0.7
SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.4
cosine_similarity=tf.keras.metrics.CosineSimilarity()

#empty the buffers
distances=[]    
faces_cropped=[]

#load ref embeddings
ref_embeddings=np.load('ref_embeddings.npy')

#load ref names dict
with open('ref_embeddings_names.pkl', 'rb') as f:
    ref_embeddings_names = pickle.load(f)

# define a video capture object
vid = cv2.VideoCapture(0)

#define a timer to measure the performance
start_time=time.time()

while(True):
    #empty the buffers
    distances=[]    
    faces_cropped=[]

    # Capture the video frame
    # by frame    
    ret, frame = vid.read()
    
    embeddings,box=calc_embs(frame)
    x,y,w,h=box
    
    for j in range(len(ref_embeddings)):
        dist=calc_dist(embeddings,ref_embeddings[j])
#                 dist = scipy.spatial.distance.euclidean(ref_embeddings[j], embeddings)
        distances.append(dist)
#                 cosine=cosine_similarity(ref_embeddings[j], embeddings)
#                 cosines.append(cosine)

            #choose the minimum distance and user
        thersold_value=1.1
#             thersold_value_cosine=0.5
        indexes=np.argmin(distances)
#             indexes=np.argmax(cosines)            
#             print(len(cosines))
#             decision_value=distances[indexes[0]]
#             decision_value=cosines[indexes]
        decision_value=distances[indexes]    
        if decision_value > thersold_value:
            name='Unknown'
        else:      
            name=ref_embeddings_names[indexes]
            
    #draw the frame
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,)          
    cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)

    #measure the performance
    cv2.putText(frame, str(int(1/(time.time()-start_time))), (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
    start_time=time.time()
    print(distances)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


# In[7]:


plt.imshow(dummy2)


# In[ ]:




