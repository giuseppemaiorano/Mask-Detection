#!/usr/bin/env python
# coding: utf-8

# In[63]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


# In[64]:


prototxtPath=os.path.sep.join([r'C:\Users\giuse\Downloads\face-mask-detector\face_detector','deploy.prototxt'])


# In[65]:


weightsPath=os.path.sep.join([r'C:\Users\giuse\Downloads\face-mask-detector\face_detector','res10_300x300_ssd_iter_140000.caffemodel'])


# In[66]:


net=cv2.dnn.readNet(prototxtPath,weightsPath)


# In[67]:


model=load_model(r'C:\Users\giuse\Downloads\face-mask-detector\mobilenet_v2.model')


# In[78]:


image=cv2.imread(r'C:\Users\giuse\Pictures\pigi2.JPG')


# In[79]:


image


# In[80]:


(h,w)=image.shape[:2]


# In[81]:


blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))


# In[82]:


net.setInput(blob)
detections=net.forward()


# In[83]:


for i in range(0,detections.shape[2]):
    confidence=detections[0,0,i,2]
    
    
    if confidence>0.5:
        box=detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype('int')
        
        (startX,startY)=(max(0,startX),max(0,startY))
        (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
        
        face=image[startY:endY, startX:endX]
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face=cv2.resize(face,(224,224))
        face=img_to_array(face)
        face=preprocess_input(face)
        face=np.expand_dims(face,axis=0)
        
        (mask,withoutMask)=model.predict(face)[0]
        
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        
        cv2.putText(image,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(image,(startX,startY),(endX,endY),color,2)
        
        
        
cv2.imshow("OutPut",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




