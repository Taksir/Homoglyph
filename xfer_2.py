# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:47:44 2017

@author: Sam
"""
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import random

# path to the model weights files.
weights_path = r'D:/vgg16_weights.h5'
top_model_weights_path = r'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 75, 150

train_data_dir = r'../DataSet/Train'
validation_data_dir = r'../DataSet/Test'
nb_train_samples = 1760*2
nb_validation_samples = 384*2
epochs = 2
batch_size = 32
#random.seed(2000) best so far
#random.seed(3003) 90% in 20 epochs with ADAM
#random.seed(2000)


# build the VGG16 network
#model = applications.VGG16(weights='imagenet', include_top=False,
#                           input_shape=(150,150,3))

#input_tensor = Input(shape=(150,150,3))
base_model = applications.VGG16(weights='imagenet',include_top= False,
                     input_shape = (150,75,3)) #150 75 order chng korte hyse
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.70))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights('bottleneck_model.h5')
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
print('Model loaded, layer 15')

# build a classifier model to put on top of the convolutional model
#top_model = Sequential()
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning

#top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

#filepath="best_weight_1700_400_transfer_learning.hdf5"
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
plot_result= TensorBoard(log_dir='Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,plot_result]

# fine-tune the model
history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    callbacks= callbacks_list,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples//batch_size)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()