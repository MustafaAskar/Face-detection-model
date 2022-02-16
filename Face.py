#!/usr/bin/env python
# coding: utf-8

# In[11]:


#import libraries
import tensorflow as tf
#import mtcnn
import numpy as np
import cv2
import copy
import PIL
import pickle
import time 


# In[8]:


#load trained Caffe model
modelFile = "yolov3-tiny.weights"
configFile = "yolov3-tiny.cfg"
net = cv2.dnn.readNetFromDarknet(configFile, modelFile)


# In[31]:


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
    distances=[]    
    faces_cropped=[]

    # Capture the video frame
    # by frame    
    ret, frame = vid.read()

    #load image
    image_np=np.asarray(frame)

    #keep the original height and width, Caffe model require resizing to 300*300
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))

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
            x, y, x1, y1=abs(x), abs(y), abs(x1), abs(y1)
            faces_cropped.append(image_np[y:y1,x:x1])
            cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),2,)    
    
    if(len(faces_cropped)>0):
        for i in range(len(faces_cropped)):
            #preprocess the image
            #resize
            #opencv can't read from numpy direct so we convert to pil  
            dummy=PIL.Image.fromarray((faces_cropped[i]*255).astype(np.uint8))
            dummy=dummy.resize((160,160))
            faces_cropped[i]=np.asarray(dummy)

            #cast as float
            faces_cropped[i]=faces_cropped[i].astype('float32')

            #standraliztion
            mean=np.mean(faces_cropped[i])
            std=np.std(faces_cropped[i])
            faces_cropped[i]=(faces_cropped[i]-mean)/std

            #expand batch dimension so tensorflow can accept it
            faces_cropped[i]=np.expand_dims(faces_cropped[i],axis=0)

            #predict the embeddings
            results=FaceNet.predict(faces_cropped[i])
            embeddings=results

            #calculate distances
            for j in range(len(ref_embeddings)):
                dist = np.square(np.linalg.norm(ref_embeddings[j] - embeddings))
                distances.append(dist)

            #choose the minimum distance and user
            thersold_value=35
            indexes=np.argsort(distances)
            decision_value=distances[indexes[0]]
            if decision_value > thersold_value:
                name='Unknown'
            else:      
                minimum_index=indexes[0]/len(faces_cropped)
                name=ref_embeddings_names[minimum_index]
            cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)

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




