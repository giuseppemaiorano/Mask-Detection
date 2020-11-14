#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils


# In[3]:


def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w)=frame.shape[0:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()
    
    
    faces=[]
    locs=[]
    preds=[]
    
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
    
        if confidence>0.5:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=12)
        
        return (locs,preds)


# In[4]:


prototxtPath=os.path.sep.join([r'C:\Users\giuse\Downloads\face-mask-detector\face_detector','deploy.prototxt'])


# In[5]:


weightsPath=os.path.sep.join([r'C:\Users\giuse\Downloads\face-mask-detector\face_detector','res10_300x300_ssd_iter_140000.caffemodel'])


# In[6]:


faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)


# In[7]:


maskNet=load_model(r'C:\Users\giuse\Downloads\face-mask-detector\mobilenet_v2.model')


# In[8]:


vs=VideoStream(src=0).start()

while True:
    
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    
    (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
    
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY)=box
        (mask,withoutMask)=pred
        
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    
    if key==ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()


# In[ ]:




