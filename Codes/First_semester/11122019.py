from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
import keras
import os

file_path = os.path.dirname(os.path.abspath(__file__))

resnet = ResNet50(include_top=False, weights='imagenet',
                   input_shape=(224, 224, 3))

output = resnet.layers[-1].output
output = keras.layers.Flatten()(output)
resnet = Model(resnet.input, output=output)
for layer in resnet.layers:
    layer.trainable = False

model = Sequential()
model.add(resnet)
number_of_classes = 4
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(optimizer='adam', lr=0.0001,loss='categorical_crossentropy',
              metrics=['accuracy'])
#model.summary()
sgd with momentum
tune the learning rate
use vgg16
lower the bs
do augmentation
don't freeze the first layers

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    file_path + "/dataset/train",
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='categorical')
#shuffle = True

validation_generator = data_generator.flow_from_directory(
    file_path + "/dataset/val",
    target_size=(image_size, image_size),
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    validation_data=validation_generator,
    validation_steps= 2)

