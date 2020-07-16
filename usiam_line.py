
from __future__ import absolute_import
from __future__ import print_function

# -*- coding: utf-8 -*-
import numpy as np
import random
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, merge,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam,SGD,RMSprop
import os
from keras.models import Model, load_model, Sequential
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.metrics import accuracy_score as accuracy
import cv2
from keras import backend as K
from keras.regularizers import l2
from random import shuffle
import make_pairs as mp
from keras.callbacks import Callback
import vis_prediction_line as vp


os.environ["CUDA_VISIBLE_DEVICES"]="2"
input_shape=(200,200,1)
train_set='ahte_train_test'
train_set_size=30000
validation_set='ahte_train_test'
validation_set_size=3000
version='11'
learning_rate=0.00001
epochs=100
batch_size=32

continue_from_best=False
continue_from_version=''

def supervised_loaddata(folderName):
    np.random.seed(0)
    random.seed(0)
    pairs = []
    labels = []
    for page in sorted(os.listdir(folderName)):
        x=[]
        y=[]
        label=0
        for textclass in sorted(os.listdir(folderName+'/'+page)):
            kota=0
            for filename in sorted(os.listdir(folderName+'/'+page+'/'+textclass)):
                image=cv2.imread(folderName+'/'+page+'/'+textclass+'/'+filename,0)
                x.append(image)
                y.append(label)
                kota=kota+1
                if kota>=patches_per_book:
                    break
            label=label+1
        ax=np.array(x)
        ay=np.array(y)
        ax = ax.reshape(ax.shape[0],ax.shape[1],ax.shape[2],1)
        ax = ax.astype('float32')
        ax /= 255.
        numberOfSubwords=len(np.unique(ay))
        indices = [np.where(ay == i)[0] for i in range(numberOfSubwords)]
        for d in range(numberOfSubwords):
            numberOfSamples=len(indices[d])
            for i in range(numberOfSamples):
                for j in range (1, numberOfSamples-i):
                    z1, z2 = indices[d][i], indices[d][i+j]
                    pairs += [[ax[z1], ax[z2]]]
                    r = random.randrange(1, numberOfSubwords-1)
                    ri = (d + r) % (numberOfSubwords)
                    s=random.randrange(0,len(indices[ri]))
                    z1, z2 = indices[d][i], indices[ri][s]
                    pairs += [[ax[z1], ax[z2]]]
                    labels += [1,0]
    apairs=np.array(pairs)
    alabels=np.array(labels)
    ind = [i for i in range(alabels.shape[0])]
    shuffle(ind)
    apairs =apairs[ind,:,:,:,:]
    alabels = alabels[ind,]
    return apairs,alabels

def unsupervised_loaddata(folderName, set_size):
    pairs = []
    labels = []
    for i in range (set_size):
        print(i)
        p1, p2, label = mp.get_random_pair(folderName)
        p1=p1.reshape(p1.shape[0],p1.shape[1],1)
        p2=p2.reshape(p2.shape[0],p2.shape[1],1)
        pairs += [[p1, p2]]
        labels += [label]

    apairs=np.array(pairs)
    print(apairs.shape)
    alabels=np.array(labels)
    return apairs,alabels

def create_base_network(input_dim):
    inputs = Input(shape=input_dim)
    conv_1=Conv2D(64,(5,5),padding="same",activation='relu',name='conv_1')(inputs)
    conv_1=MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2=Conv2D(128,(5,5),padding="same",activation='relu',name='conv_2')(conv_1)
    conv_2=MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3=Conv2D(256,(3,3),padding="same",activation='relu',name='conv_3')(conv_2)
    conv_3=MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_4')(conv_3)
    conv_5=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_5')(conv_4)
    conv_5=MaxPooling2D(pool_size=(2, 2))(conv_5)

    dense_1=Flatten()(conv_5)
    dense_1=Dense(512,activation="relu")(dense_1)
    dense_1=Dropout(0.5)(dense_1)
    dense_2=Dense(512,activation="relu")(dense_1)
    dense_2=Dropout(0.5)(dense_2)
    return Model(inputs, dense_2)


tr_pairs, tr_y=unsupervised_loaddata(train_set, train_set_size)
te_pairs, te_y=unsupervised_loaddata(validation_set, validation_set_size)

print('number of train pairs:', len(tr_pairs))
print('number of validation pairs:', len(te_pairs))

if (continue_from_best):
    model=load_model('bestmodel'+continue_from_version)
else:
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    fc6=concatenate([processed_a, processed_b])
    fc7=Dense(1024, activation = 'relu')(fc6)
    fc8=Dense(1024, activation = 'relu')(fc7)
    fc9=Dense(1, activation='sigmoid')(fc8)
    model = Model([input_a, input_b], fc9)
    model.summary()

mcp = ModelCheckpoint('bestmodel'+version, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
logs = CSVLogger('log'+version)



class Visual(Callback):
    def on_epoch_end(self, epoch,logs=None):
        vp.pca(version)


myVisual=Visual()
adam=Adam(lr=learning_rate)
#sgd = SGD(lr=learning_rate)
#rms = RMSprop(lr=learning_rate)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          callbacks=[mcp,logs,myVisual])

del model
model=load_model('bestmodel'+version)

#y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
#tr_acc = accuracy(tr_y, y_pred.round())
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = accuracy(te_y, y_pred.round())
#print('* Accuracy on training set: %0.4f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.4f%%' % (100 * te_acc))
