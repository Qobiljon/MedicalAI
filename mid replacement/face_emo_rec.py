# importing libraries
import pandas as pd
import numpy as np
import time as t
import cv2 as cv
import glob
import collections
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

np.set_printoptions(threshold=np.inf)

# loading dataset
data = pd.read_csv('fer2013.csv')
print(data.values.shape)

# getting images and labels
x = data['pixels']
y = data['emotion']
X = np.array(list(map(str.split, x)), np.float32)

'''
#displaying images
for ix in range(1,401):
    plt.axis('off')
    plt.subplot(20,20,ix)
    plt.imshow(X[ix-1+28709].reshape((48, 48)), interpolation='none', cmap='gray')
plt.axis('off')
plt.show()
'''
# garbage = X
# plt.imshow(X[59].reshape((48, 48)), interpolation='none', cmap='gray')
# plt.show()

# data splitting
x_train = X[:28709]
y_train = y[:28709]
x_test = X[28709:]
y_test = y[28709:]
y_class = y_test
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
input_shape = (48, 48, 1)
x_train /= 255
x_test /= 255

# print("check train: ", y_train.value_counts())
# print("check test: ", y_test.value_counts())

# validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=101)
y_train = np_utils.to_categorical(y_train, 7)
y_val = np_utils.to_categorical(y_val, 7)
y_test = np_utils.to_categorical(y_test, 7)
print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_val', x_val.shape)
print('y_val', y_val.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)

# data generator
datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest')
# testgen = ImageDataGenerator(rescale=1./255)
datagen.fit(x_train)

# checking generated images
'''
for ix in range(1,17):
    plt.axis('off')
    plt.subplot(4,4,ix)
    plt.imshow(x_train[ix].reshape((48, 48)), interpolation='none', cmap='gray')
plt.axis('off')
plt.show()
'''
# batches = datagen.flow(x_train[1:17], y_train[1:17], batch_size=64)
# x_batch, y_batch = next(batches)
# print('x_batch: ', x_batch.shape)
'''
for ix in range(1,16):
    plt.axis('off')
    plt.subplot(4,4,ix)
    plt.imshow(x_batch[ix].reshape((48, 48)), interpolation='none', cmap='gray')
plt.axis('off')
plt.show()
'''
# print('datagen', datagen)


# '''
# building a CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same", activation='relu', name='b_conv1'))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu', name='b_conv2'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding="same", name='b_conv3'))
model.add(Conv2D(128, (3, 3), padding="same", activation='relu', name='b_conv4'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu', padding="same", name='b_conv5'))
model.add(Conv2D(256, (3, 3), padding="same", activation='relu', name='b_conv6'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), activation='relu', padding="same", name='b_conv7'))
model.add(Conv2D(512, (3, 3), padding="same", activation='relu', name='b_conv8'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', name='b_dense1'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# compiling the model
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=2)


# preparing hyperparameters for training
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.0001, patience=10, verbose=1)
batch_size = 128

print("\n\n---------starting training\n\n")
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=len(x_train) // batch_size, callbacks=[lr_reduce],
                              validation_data=(x_val, y_val),
                              epochs=250, verbose=2)
print("\n\n---------end of training\n\n")

# save model if needed
model.save('fer2013_vgg_b.h5')

# testing the model
loss = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
print('Accuracy: ', loss[1], '\n')
# '''


# '''
# load and finetune the pre-trained model (0.6% accuracy increase)
model2 = load_model('fer2013_vgg_b.h5')
loss = model2.evaluate(x_test, y_test, batch_size=128, verbose=2)
print('\nAccuracy: ', loss, '\n\n\n')

print(model2.summary())
opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=0.0001, patience=6, verbose=1)
model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=40, batch_size=128, verbose=2, callbacks=[lr_reduce])
model2.save('fer2013_vgg_b_finetuned.h5')
loss = model2.evaluate(x_test, y_test, batch_size=128, verbose=2)
print('\nAccuracy after finetuning: ', loss, '\n\n\n')
# '''


# '''
model2 = load_model('fer2013_vgg_b_finetuned.h5')
loss = model2.evaluate(x_test, y_test, batch_size=128, verbose=2)
print('\nAccuracy: ', loss, '\n\n\n')

pred = model2.predict_classes(x_test)
# y_classes = pred.argmax(axis=-1)
print('Done.\nAccuracy: %f' % accuracy_score(y_class, pred))

# print("actual: ", y_class.value_counts())
# print("predicted: ", collections.Counter(pred))
