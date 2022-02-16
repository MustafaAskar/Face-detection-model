#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import tensorflow as tf
#import mtcnn
import numpy as np
import cv2
import copy
import PIL
import pickle


# In[11]:


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
    
while(True):
      
    #empty the buffers
    distances=[]    
    faces_cropped=[]
    
    # Capture the video frame
    # by frame    
    ret, frame = vid.read()

    #load image
    image_np=np.asarray(frame)
    image_np_copy=copy.deepcopy(image_np)

    #keep the original height and width, Caffe model require resizing to 300*300
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))

    #path the image to the model
    net.setInput(blob)

    #extract
    faces = net.forward()

    #crop faces from the image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces_cropped.append(image_np[y:y1,x:x1])
                modified_imag=cv2.rectangle(image_np_copy,(x,y),(x1,y1),(255,0,0),2,)    
    if(len(faces_cropped)==0):
        # Display the original frame
        continue
    
    #preprocess the image
    #resize
    for i in range(len(faces_cropped)):
      faces_cropped[i]=PIL.Image.fromarray((faces_cropped[i]*255).astype(np.uint8))
      faces_cropped[i]=faces_cropped[i].resize((160,160))
      faces_cropped[i]=np.asarray(faces_cropped[i])

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
        dist = np.linalg.norm(ref_embeddings[j] - embeddings)
        distances.append(dist)
    print(distances[i])
    
    #choose the minimum distance and user
    thersold_value=7.5
    indexes=np.argsort(distances)
    decision_value=distances[indexes[0]]
    if decision_value > thersold_value:
      name='Unknown'
    else:      
      minimum_index=indexes[0]/len(faces_cropped)
      name=ref_embeddings_names[minimum_index]
    cv2.putText(modified_imag, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)
      
    # Display the resulting frame
    cv2.imshow('frame', modified_imag)
      
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




