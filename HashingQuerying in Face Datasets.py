#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import cv2 


# In[3]:


image = cv2.imread('color.jpeg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
  
plt.imshow(image)


# In[4]:


pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)


# In[6]:


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()] 
segmented_image = segmented_data.reshape((image.shape)) 
plt.imshow(segmented_image)


# In[ ]:




