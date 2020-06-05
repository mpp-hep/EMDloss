
# coding: utf-8

# In[2]:


#import setGPU
import os
import numpy as np
import h5py
import glob
import itertools
import sys
from sklearn.utils import shuffle

import setGPU

# In[12]:

penaltyValue = float(sys.argv[1])

fileIN = "/data/mpierini/qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND_List.h5"
f = h5py.File(fileIN, 'r')
X_train = np.array(f.get("X_train"), dtype=np.float32)
Y_train = np.array(f.get("Y_train"), dtype=np.float32)
X_val = np.array(f.get("X_val"), dtype=np.float32)
Y_val = np.array(f.get("Y_val"), dtype=np.float32)
print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)

# keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten, Concatenate, Reshape, BatchNormalization, Activation, Lambda
from keras.layers import MaxPooling2D
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


# In[ ]:


image_shape = (X_train.shape[-3],X_train.shape[-2],X_train.shape[-1])

# In[ ]:

inputImage = Input(shape=(image_shape))
x = BatchNormalization()(inputImage)
#
x = Conv2D(10, kernel_size=(6,1), data_format="channels_first", strides=(1, 1), padding="same", 
            input_shape=image_shape, kernel_initializer='lecun_uniform')(x)
#x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
#
x = Conv2D(5, kernel_size=(10,1), data_format="channels_first", strides=(1, 1), padding="same", 
            input_shape=image_shape, kernel_initializer='lecun_uniform')(x)
#x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
#
x = Conv2D(2, kernel_size=(5,1), data_format="channels_first", strides=(1, 1), padding="same", 
            input_shape=image_shape, kernel_initializer='lecun_uniform')(x)
#x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
#
x = Flatten()(x)
x = Dense(80, activation="relu", kernel_initializer='lecun_uniform', name='dense_relu1')(x)
#x = Dropout(0.2)(x)
#
x = Dense(40, activation="relu", kernel_initializer='lecun_uniform', name='dense_relu2')(x)
#x = Dropout(0.2)(x)
#
x = Dense(20, activation="relu", kernel_initializer='lecun_uniform', name='dense_relu3')(x)
#x = Dropout(0.2)(x)
#
# CUSTOM
# -------
#outputMean = Dense(1, activation='linear', kernel_initializer='lecun_uniform')(x)
#outputSigma = Dense(1, activation='elu', kernel_initializer='lecun_uniform')(x)
#outputSigma = Lambda(lambda x: x + 1.000001)(outputSigma)
#output = Concatenate()([outputMean,outputSigma])
# MSE or MAE
# ----------
output = Dense(1, activation='linear', kernel_initializer='lecun_uniform')(x)   
model = Model(inputs=inputImage, outputs=output)

from keras import backend as K
# Gaussian distributed variables
def chisqLoss(x, pars):
    mu = pars[:,0]
    sigma = pars[:,1]
    norm_x = K.tf.divide(x - mu, sigma)
    nll_loss = K.log(sigma) + 0.5*K.square(norm_x)
    nll_loss = K.mean(nll_loss, axis=-1)
    return nll_loss

# In[ ]:

def MAE_AsyLoss(x, xhat):
    loss = K.abs(x-xhat)
    #penalty = (K.sign(x-xhat)+1)/2.*K.abs(x-xhat)/x
    penalty = (K.sign(x-xhat)+1)/2.*penaltyValue*loss
    return loss*(1+penalty)

#model.compile(optimizer='adam', loss=chisqLoss)
#model.compile(optimizer='adam', loss='mae')
#model.compile(optimizer='adam', loss='mape')
model.compile(optimizer='adam', loss=MAE_AsyLoss)
model.summary()

# In[ ]:


# train 
history = model.fit(X_train, Y_train, epochs=500, batch_size=128, verbose = 2,
                  validation_data=(X_val, Y_val),
                 callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
                TerminateOnNaN()])

#nameModel = 'EMD_Dense_MAE'
#nameModel = 'EMD_Dense_MAPE'
nameModel = 'EMD_Dense_MAE_AsymmetryLarge_%s' %sys.argv[1]
# store history
f = h5py.File("models/%s_history.h5" %nameModel, "w")
f.create_dataset("training_loss", data=np.array(history.history['loss']),compression='gzip')
f.create_dataset("validation_loss", data=np.array(history.history['val_loss']),compression='gzip')
f.close()

# store model
model_json = model.to_json()
with open("models/%s.json" %nameModel, "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/%s.h5" %nameModel)
