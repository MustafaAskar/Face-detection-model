#!/usr/bin/env python
# coding: utf-8

# In[47]:


#import libraries
import numpy as np
import cv2
import PIL
import pickle
import os
import matplotlib.pyplot as plt
################
import mediapipe as mp
###########################
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import matplotlib.pyplot as plt


# In[48]:


#load the image pathes
images_path=os.listdir('ref images')


# In[49]:


#load the model
app = FaceAnalysis(name='buffalo_sc',providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))


# In[50]:


if os.path.exists('ref_embeddings_names.pkl'):
    #load ref names dict
    with open('ref_embeddings_names.pkl', 'rb') as f:
        ref_embeddings_names = pickle.load(f)
else:
    ref_embeddings_names={}
    
if os.path.exists('ref_embeddings.npy'):
    #load ref embeddings
    embeddings=np.load('ref_embeddings.npy')
else:
    embeddings=[]


# In[51]:


#load the images
base_dir='ref images'
for i,image_path in enumerate(images_path):
    image_name=os.path.join(base_dir,image_path)
    person_name=image_path.split('.')[0]
    print(image_name)
    img = cv2.imread(image_name)
    faces = app.get(img)
    rimg = app.draw_on(img, faces)
    plt.imshow(rimg)
    plt.show()
    if person_name in ref_embeddings_names.values():
        print('name already exists, skipping')
        continue
    ref_embeddings_names[i]=(person_name)
    embeddings.append(faces[0].embedding)


# In[52]:


#sanitiy check
print(len(embeddings))
print(embeddings[0].shape)


# In[53]:


#print dictionary with names
print(ref_embeddings_names)


# In[54]:


#save the embeddings
np.save('ref_embeddings',embeddings)

with open('ref_embeddings_names.pkl', 'wb') as f:
    pickle.dump(ref_embeddings_names, f)


# In[55]:


print(embeddings[0].shape)


# In[56]:


print(embeddings[0])

