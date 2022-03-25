#!/usr/bin/env python
# coding: utf-8

# In[9]:


#import libraries
from scipy.spatial import distance
import pickle
###########################
import mediapipe as mp
##########################
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time


# In[10]:


# load mediapipe 
mp_facedetector = mp.solutions.face_detection
detector=mp_facedetector.FaceDetection(min_detection_confidence=0.7)

#load insight project
app = FaceAnalysis(name='buffalo_sc',providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

#load ref embeddings
ref_embeddings=np.load('ref_embeddings.npy')

#load ref names dict
with open('ref_embeddings_names.pkl', 'rb') as f:
    ref_embeddings_names = pickle.load(f)


# In[11]:


#performance and accuracy configurations
refresh_period=1
thersold_value=26

# define a video capture object
vid = cv2.VideoCapture(0)

#start the program
while(True):
    #empty the buffers
    names_tracking=[]
    
    # Capture the video frame by frame    
    _, frame = vid.read()
    
    #get the box and embeddings
    faces=app.get(frame)
    
    if faces:
        for i in range(len(faces)):
            #select a face
            face=faces[i]
            #empty the buffer
            distances=[]
            #calculate the distance
            for ref_embedding in ref_embeddings:
                dist=distance.euclidean(face.embedding,ref_embedding)
                distances.append(dist)
            print(distances)
            #choose the minimum distance and user
            indexes=np.argmin(distances)
            decision_value=distances[indexes]
            #decide which user or unknown
            if decision_value > thersold_value:
                name='Unknown'
                names_tracking.append(name)
            else:      
                name=ref_embeddings_names[indexes]
                names_tracking.append(name)
                
#             #draw the frame
#             frame = app.draw_on(frame, faces)
#             cv2.putText(frame, name, (int(face.bbox[0])+20, int(face.bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2)

#     #measure the performance
#     cv2.putText(frame, str(int(1/(time.time()-start_time))), (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
#     start_time=time.time()

#     #Display the resulting frame
#     cv2.imshow('frame', frame)
    
    #track the faces
    entry_names=len(names_tracking)
    #start the software timer
    software_timer=time.time()
    while(True):
        #empty the buffers
        nms_boxes=[]
        #measure the performance
        start_time=time.time()
        #detection using mediapipe
        #capture frame
        _, frame = vid.read()
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and find faces
        results = detector.process(image)
        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for id, detection in enumerate(results.detections):
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                x,y,w,h=abs(int(bBox.xmin * w)), abs(int(bBox.ymin * h)-30), abs(int(bBox.width * w)), abs(int(bBox.height * h)+30)
                nms_boxes.append([x, y, int(w), int(h)])
        if entry_names!=len(nms_boxes) or (time.time()-software_timer)>refresh_period:
            print('breaked')
            break
        else:
            for i in range(len(nms_boxes)):
                x, y = nms_boxes[i][0], nms_boxes[i][1]
                w, h = nms_boxes[i][2], nms_boxes[i][3]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,)          
                cv2.putText(frame, names_tracking[i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)
        #measure the performance
        cv2.putText(frame, str(int(1/(time.time()-start_time))), (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
        #Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # the 'q' button is set as the
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


# In[ ]:




