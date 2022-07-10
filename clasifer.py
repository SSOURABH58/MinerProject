import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping
import pandas as pd

from numpy.random import seed
seed(1337)
from tensorflow import random
# set_random_seed(42)
random.set_seed(42)
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, Model, optimizers

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

# train_data_dir = "data/train"
# val_data_dir = "data/val"
# test_data_dir = "data/test"

# data = "color"
data = "color"

category_names = sorted(os.listdir(data))
nb_categories = len(category_names) # number of category --
img_pr_cat = []
for category in category_names:
    folder = data + '/' + category
    img_pr_cat.append(len(os.listdir(folder)))
print(category_names)
sns.barplot(y=category_names, x=img_pr_cat).set_title("Number of training images per category:")

fig = plt.figure(figsize=(10, 7))
#
rows = int(nb_categories*(9/16))
columns = nb_categories - rows

count=-1
for subdir, dirs, files in os.walk(data):
    count+=1
    for i,file in enumerate(files):
        img_file = subdir + '/' + file
        image = load_img(img_file)
        # plt.figure()
        print(count)
        fig.add_subplot(rows, columns, count)
        plt.axis('off')
        plt.title(subdir)
        plt.imshow(image)
        break



# img_height, img_width = 256,256
# conv_base = vgg16.VGG16(weights='imagenet', include_top=False, pooling='max', input_shape = (img_width, img_height, 3))
#
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Dense(nb_categories, activation='softmax'))
# model.summary()

#Number of images to load at each iteration
batch_size = 8
# only rescaling
# datagen =  ImageDataGenerator(
#     rescale=1./255,validation_split=0.2
# )
# these are generators for train/test data that will read pictures #found in the defined subfolders of 'data/'
# print('Total number of train images :')
#
# train_generator = datagen.flow_from_directory(
# directory=data,
# target_size = (img_height, img_width),
# batch_size = batch_size,
# subset="training",
# class_mode = "categorical")
#
# print('Total number of validation images :')
#
# val_generator = datagen.flow_from_directory(
# directory=data,
# target_size = (img_height, img_width),
# batch_size = batch_size,
# subset="validation",
# class_mode = "categorical")


# image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

# train_dataset = image_generator.flow_from_directory(batch_size=32,
#                                                  directory='full_dataset',
#                                                  shuffle=True,
#                                                  target_size=(280, 280), 
#                                                  subset="training",
#                                                  class_mode='categorical')

# validation_dataset = image_generator.flow_from_directory(batch_size=32,
#                                                  directory='full_dataset',
#                                                  shuffle=True,
#                                                  target_size=(280, 280), 
#                                                  subset="validation",
#                                                  class_mode='categorical')


# Fun stuff -------------------------------

# learning_rate = 5e-5
# epochs = 10
#
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # sess = tf.Session(config=config)
# checkpoint_path = "saves/"
# print(os.getcwd())
# model = tf.keras.models.load_model(checkpoint_path,compile=False)
# # directory_checkpoint = os.path.dirname(checkpoint_path)
# # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
#
# early_stopping = EarlyStopping(monitor='val_accuracy')
# checkpoint = ModelCheckpoint(checkpoint_path+"modle_{epoch:02d}", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, save_freq='epoch',mode='auto')
# model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate, clipnorm=1.), metrics=['accuracy'])
# # model.load_weights(checkpoint_path)
# history = model.fit(train_generator,
#                               epochs=epochs,
#                               shuffle=False,
#                               validation_data=val_generator,
#                               callbacks=[checkpoint]
#                               )
# print("done")
# model.save('model_H5.h5')
# model.save('sign_classifier.h5')
# print('Model Saved!')
# history = model.fit(...)
# for key in history.history:
#     print(key)

# model = models.load_model("sign_classifier.h5")

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1,len(acc)+1)
# plt.figure()
# plt.plot(epochs, acc, 'b', label = 'Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.savefig('Accuracy.jpg')
# plt.figure()
# plt.plot(epochs, loss, 'b', label = 'Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig('Loss.jpg')

plt.show()


# color/Apple___Apple_scab
# color/Apple___Black_rot
# color/Apple___Cedar_apple_rust
# color/Apple___healthy
# color/Blueberry___healthy
# color/Cherry_(including_sour)___healthy
# color/Cherry_(including_sour)___Powdery_mildew
# color/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
# color/Corn_(maize)___Common_rust_
# color/Corn_(maize)___healthy
# color/Corn_(maize)___Northern_Leaf_Blight
# color/Grape___Black_rot
# color/Grape___Esca_(Black_Measles)
# color/Grape___healthy
# color/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
# color/Orange___Haunglongbing_(Citrus_greening)
# color/Peach___Bacterial_spot
# color/Peach___healthy
# color/Pepper,_bell___Bacterial_spot
# color/Pepper,_bell___healthy
# color/Potato___Early_blight
# color/Potato___healthy
# color/Potato___Late_blight
# color/Raspberry___healthy
# color/Soybean___healthy
# color/Squash___Powdery_mildew
# color/Strawberry___healthy
# color/Strawberry___Leaf_scorch
# color/Tomato___Bacterial_spot