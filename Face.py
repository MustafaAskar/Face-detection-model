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
from scipy import stats


# In[6]:


#empty the buffers
distances=[]    
faces_cropped=[]

#load trained Caffe model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#load FaceNet model
FaceNet=tf.keras.models.load_model('facenet_keras.h5', compile=False)

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
    faces_cropped=[]
    boxes=[]
    names=[]
    
    # Capture the video frame
    # by frame    
    ret, frame = vid.read()

    #load image
    image_np=np.asarray(frame)

    #keep the original height and width, Caffe model require resizing to 300*300
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(image=frame,size=(300, 300))

    #pass the image to the model
    net.setInput(blob)

    #extract
    faces = net.forward()

    #crop faces from the image
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            x, y, x1, y1=abs(x)+1, abs(y)+1, abs(x1)+1, abs(y1)+1
            faces_cropped.append(image_np[y:y1,x:x1])
            boxes.append([x,y,x1,y1])
    
    if(len(faces_cropped)>0):
        for i in range(len(faces_cropped)):
            #preprocess the image
            #resize
            #opencv can't read from numpy direct so we convert to pil  
            dummy=PIL.Image.fromarray(faces_cropped[i])
            dummy=dummy.resize((160,160))
            faces_cropped[i]=np.asarray(dummy)

            #cast as float
            faces_cropped[i]=faces_cropped[i].astype('float32')

            #standraliztion
            faces_cropped[i]=stats.zscore(faces_cropped[i],axis=None)

            #expand batch dimension so tensorflow can accept it
            faces_cropped[i]=np.expand_dims(faces_cropped[i],axis=0)

            #predict the embeddings
            results=FaceNet.predict(faces_cropped[i])
            embeddings=results
            
            #empty the distance buffer
            distances=[]
            
            #calculate distances
            for j in range(len(ref_embeddings)):
                dist = np.square(np.linalg.norm(ref_embeddings[j] - embeddings))
                distances.append(dist)

            #choose the minimum distance and user
            thersold_value=140
            indexes=np.argsort(distances)
            decision_value=distances[indexes[0]]
            if decision_value > thersold_value:
                name='Unknown'
                names.append(name)
            else:      
                name=ref_embeddings_names[indexes[0]]
                names.append(name)
                
        #draw the frame
        for i in range(len(faces_cropped)):         
            box=boxes[i]
            name=names[i]
            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,0,0),2,)    
            cv2.putText(frame, name, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)

    #measure the performance
    cv2.putText(frame, str(int(1/(time.time()-start_time))), (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
    start_time=time.time()

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


# In[ ]:




