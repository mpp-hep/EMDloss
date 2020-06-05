#import setGPU
import os
import numpy as np
import h5py
import glob
import itertools
import sys
from sklearn.utils import shuffle
import setGPU

inputFile = sys.argv[1]
penaltyValue = float(sys.argv[2])

f = h5py.File(inputFile, 'r')
print(f.keys())

X1_train = np.array(f.get("J1_train"), dtype=np.float32)
X1_train = np.concatenate((X1_train, np.array(f.get("J2_train"), dtype=np.float32)), axis=2)
Y_train = np.array(f.get("EMD_train"), dtype=np.float32)

X1_val = np.array(f.get("J1_val"), dtype=np.float32)
X1_val = np.concatenate((X1_val, np.array(f.get("J2_val"), dtype=np.float32)), axis=2)
Y_val = np.array(f.get("EMD_val"), dtype=np.float32)

X1_val = X1_val[:200000,:,:]
Y_val = Y_val[:200000]

X1_train = np.reshape(X1_train, (X1_train.shape[0], X1_train.shape[1], X1_train.shape[2], 1))
X1_val = np.reshape(X1_val, (X1_val.shape[0], X1_val.shape[1], X1_val.shape[2], 1))

print(X1_train.shape, Y_train.shape)
print(X1_val.shape, Y_val.shape)

# keras imports
from tensorflow.keras import models, layers, utils
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import callbacks 
from tensorflow.keras import optimizers

image_shape = (X1_train.shape[-3],X1_train.shape[-2],X1_train.shape[-1])

input1 = layers.Input(shape=(image_shape))
#
x = layers.BatchNormalization()(input1)
x = layers.Conv2D(32, kernel_size=(3,3), data_format="channels_last", strides=(1, 1), padding="valid", 
            input_shape=image_shape, kernel_initializer='lecun_uniform')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.AveragePooling2D(pool_size=(2,1))(x)
x = layers.Conv2D(16, kernel_size=(3,3), data_format="channels_last", strides=(1, 1), padding="valid",
            kernel_initializer='lecun_uniform')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.AveragePooling2D(pool_size=(2,1))(x)
x = layers.Conv2D(8, kernel_size=(3,2), data_format="channels_last", strides=(1, 1), padding="valid",
            kernel_initializer='lecun_uniform')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.AveragePooling2D(pool_size=(2,1))(x)
x = layers.Flatten()(x)
#
x = layers.Dense(80, activation="relu", kernel_initializer='lecun_uniform', name='dense_relu1')(x)
x = layers.Dropout(0.2)(x)
#
x = layers.Dense(40, activation="relu", kernel_initializer='lecun_uniform', name='dense_relu2')(x)#
x = layers.Dropout(0.2)(x)
#
x = layers.Dense(20, activation="relu", kernel_initializer='lecun_uniform', name='dense_relu3')(x)
x = layers.Dropout(0.2)(x)
#
output = layers.Dense(1, activation='linear', kernel_initializer='lecun_uniform')(x)   
model = models.Model(inputs=input1, outputs=output)

model.compile(optimizer=optimizers.Adam(), loss='mae')
model.summary()

my_callbacks = [
    callbacks.EarlyStopping(patience=10, verbose=1),
    #callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
    callbacks.TerminateOnNaN()]

# train 
history = model.fit(X1_train, Y_train, epochs=500, batch_size=128, verbose = 2,
                  validation_data=(X1_val, Y_val), callbacks = my_callbacks)

nameModel = 'EMD_Conv2D_MAE'
#nameModel = 'EMD_Dense_MAPE'
#nameModel = 'EMD_Dense_MAE_AsymmetryLarge_%s' %sys.argv[1]

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
