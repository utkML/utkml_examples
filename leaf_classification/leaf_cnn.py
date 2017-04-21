from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image

from PIL import Image
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Let's start with some simple globals
dim = 20
batch_size = 128

train = pd.read_csv('train.csv')

images = np.empty((len(train['id']), dim, dim, 1))

i = 0
for id in train['id']:
    filename = "images/"+str(id)+".jpg"
    im = Image.open(filename)
    im.thumbnail([dim, dim])
    im = np.array(im)
    height, width = im.shape
    
    #calculate destination coordinates
    h1 = int((dim - height) / 2)
    h2 = h1 + height
    w1 = int((dim - width) / 2)
    w2 = w1 + width
    
    images[i, h1:h2, w1:w2, 0] = im
    i += 1

y_all = train.pop('species')
y_all = LabelEncoder().fit(y_all).transform(y_all)
y_all = to_categorical(y_all)
train.pop('id')
featuresAll = StandardScaler().fit(train).transform(train)

image_mean = images.mean().astype(np.float32)
image_std = images.std().astype(np.float32)

def normalize(x): 
    return (x-image_mean)/image_std

model = Sequential([
    Lambda(normalize, input_shape=(dim,dim,1)),
        Conv2D(16,(3,3), activation='relu'),
        Conv2D(16, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32,(3,3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(20, activation='relu'),
        Dense(y_all.shape[1], activation='softmax')
    ])
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#split test and validation
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=0)
train_index, val_index = next(sss.split(images, y_all))
x_train, y_train = images[train_index], y_all[train_index]
features_train = featuresAll[train_index]
x_val, y_val = images[val_index], y_all[val_index]
features_val = featuresAll[val_index]

print("Training image set shape: {}".format(x_train.shape))
print("Training label shape: {}".format(y_train.shape))
print("Validation image set shape: {}".format(x_val.shape))
print("Training label shape: {}".format(y_val.shape))

generator = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, 
                                     height_shift_range=0.08, zoom_range=0.08)

train_batches = generator.flow(x_train, y_train, batch_size=batch_size)
val_batches = generator.flow(x_val, y_val, batch_size=batch_size)

model.fit_generator(train_batches, train_batches.n, nb_epoch=5, 
                    validation_data=val_batches, nb_val_samples=val_batches.n)
