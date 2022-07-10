import os

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

data = "color"

category_names = sorted(os.listdir(data))

model_path = './data/00000-test-train/model-03-0.00.hdf5'
model = load_model(model_path)

test_datagen = ImageDataGenerator(rescale=1./255)
test_path_apple = 'testdata/apple/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'
test_path_corn = 'testdata/apple/0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG'

image_path = input("image path :")
# predicting images
img = image.load_img(image_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=1)
print(classes)

#load the image
# my_image = load_img(test_path_corn, target_size=(256, 256))

#preprocess the image
# images = []
#
# for path in sorted(os.listdir(data+'/apple')):
#     my_image = load_img(data+'/apple/'+path, target_size=(256, 256))
#     my_image = img_to_array(my_image)
#     print(my_image.shape)
#     my_image = np.expand_dims(my_image, axis=0)
#     print(my_image.shape)
#     images.append(my_image)
# # my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
# # my_image = preprocess_input(my_image)
# # image = tf.image.resize(my_image, (256, 256))
# print(images)
# #make the prediction
# prediction = model.predict([images])
# predict = model.predict_generator(test_generator,steps = nb_samples)


# datagen =  ImageDataGenerator(
#     rescale=1./255
# )
# train_generator = datagen.flow_from_directory(
# directory=data,
# target_size = (256, 256),
# batch_size = 8,
# # subset="test",
# class_mode = None)

# prediction = model.predict(train_generator)

# print(train_generator)
# readible = [ category_names[int(np.where(categ == np.max(categ))[0])] for categ in prediction]
#
# print(readible)