#!/usr/bin/env python
# coding: utf-8

# # Introduction to Convolution Neural Networks

# ## Import the libraries

# In[1]:


from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
from keras.models import Sequential #give a linear stack of neural network layers
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data

# In[2]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## Pre-processing
# Our MNIST images only have a depth of 1. We must explicitly declare that.

# In[4]:


num_classes = 10
epochs = 3

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test, num_classes)


# In[5]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:





# ## Creating and compiling

# In[6]:


cnn = Sequential()


# In[7]:


#add convolution layer, first param will be the amount of filters we have
cnn.add(Conv2D(32, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='relu'))


# In[8]:


#add max pooling layer
cnn.add(MaxPooling2D())


# In[9]:


#add next convolution layer with 64 filters
cnn.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'))


# In[10]:


#add max pooling layer
cnn.add(MaxPooling2D())


# In[11]:


#flatten the network - as we have a fully connected (dense) network coming next
cnn.add(Flatten())


# In[12]:


#fully connected layer
cnn.add(Dense(1024, activation='relu'))


# In[13]:


#fully connected layer - output layer
#need 10 differnet classes - must be a softmax layer
cnn.add(Dense(10,activation='softmax'))


# In[14]:


#because the output will be for multiple digits we use categorical crossentropy
cnn.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[15]:


print(cnn.summary())


# ## Training the model

# In[16]:


#history_cnn = cnn.fit(X_train,y_train,epochs=5,verbose=1,validation_data=(X_train,y_train))


# In[17]:


#plt.plot(history_cnn.history['acc'])
#plt.plot(history_cnn.history['val_acc'])


# In[18]:


#within keras we can load in weights, only diffference is we are using pre trained weights done on an identical model
cnn.load_weights('weights/cnn-model5.h5')


# In[19]:


score = cnn.evaluate(X_test,y_test)


# ## Accuracy of the model

# In[21]:


#cnn model is more accurate that the previous neural network model
score


# In[ ]:





# In[ ]:




