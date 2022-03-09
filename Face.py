#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model


# In[2]:


#load FaceNet model
FaceNet=tf.keras.models.load_model('facenet_keras.h5',compile=False)

#load trained YOLO model
config_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny_2000.weights"
YOLO = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = YOLO.getLayerNames()
ln = [ln[i - 1] for i in YOLO.getUnconnectedOutLayers()]

#specifiy the image size
image_size = 160


# In[3]:


def prewhiten(x):
    #predefine some varabiles
    y=tf.image.per_image_standardization(x)
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    #l2 normalization
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(img,margin):
    #to fix the bug of no detection
    nms_boxes=[[0,0,0,0]]
    
    #empty the buffers
    boxes, confidences, class_ids,names, aligned_images = [], [], [], [], []

    #keep the original height and width, Caffe model require resizing to 300*300
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255, (160, 160), swapRB=True, crop=False)

    #pass the image to the model
    YOLO.setInput(blob)

    #extract
    faces = YOLO.forward(ln)
    
    # loop over each of the layer outputs
    for output in faces:
        # loop over each of the object detections
        for detection in output:
            # extract the confidence (as a probability) the current object detection
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
                x,y,width,height=abs(x)+2,abs(y)+2,abs(width)+2,abs(height)+2
                
                # update our list of bounding box coordinates, confidences
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                
    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    
    #loop over the suppressed boxes
    if(len(idxs)>0):
        nms_boxes=[]
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            #append new suppressed boxes to draw it
            nms_boxes.append([x, y, int(w), int(h)])
            face_cropped=img[y:y+h,x:x+w]
            aligned = resize(face_cropped, (image_size, image_size), mode='reflect')
            aligned_images.append(aligned)
    return np.array(aligned_images), nms_boxes

def calc_embs(img, margin=10, batch_size=1):
    cropped_images, boxes=load_and_align_images(img, margin)
    #to fix the bug of no detection
    if(cropped_images.shape[0]!=0):
        aligned_images = prewhiten(cropped_images)
        embs = []
        #calculate the emeddings
        for start in range(0, len(aligned_images), batch_size):
            pd=FaceNet.predict_on_batch(aligned_images[start:start+batch_size])
            embs.append(l2_normalize(pd))
    else:
        embs=[np.zeros((128,1))]
    return embs, boxes

def calc_dist(img0, img1):
    #calculate the distances
    return distance.euclidean(img0, img1)


# In[4]:


#pre definied varaiables
CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

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
    # Capture the video frame by frame    
    ret, frame = vid.read()
    
    #calculate the embeddings and boxes
    embeddings,boxes=calc_embs(frame)
    
    #loop over every embedding to calculate the distances
    for i,embedding in enumerate(embeddings):
        distances=[]
        for j in range(len(ref_embeddings)):
            dist=calc_dist(embedding,ref_embeddings[j])
            distances.append(dist)

            #choose the minimum distance and user
            thersold_value=1
            indexes=np.argmin(distances)
            decision_value=distances[indexes]    
            if decision_value > thersold_value:
                name='Unknown'
            else:      
                name=ref_embeddings_names[indexes]

            #draw the frame
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,)          
            cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)

    #measure the performance
    cv2.putText(frame, str(int(1/(time.time()-start_time))), (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
    start_time=time.time()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


# In[5]:


plt.imshow(dummy2)


# In[ ]:




