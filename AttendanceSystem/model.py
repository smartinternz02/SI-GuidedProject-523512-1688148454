from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders

input_folder = r'D:\face'
output = r'D:\split'

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(0.8, 0.1, 0.1))

train_path = r'D:\split\train'
test_path = r'D:\split\test'

train_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1. / 255)

train = train_gen.flow_from_directory(train_path, target_size=(224, 224), batch_size=22, class_mode='categorical')
test = test_gen.flow_from_directory(test_path, target_size=(224, 224), batch_size=22, class_mode='categorical')

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(train.num_classes, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, validation_data=test, epochs=10, steps_per_epoch=len(train), validation_steps=len(test))

model.save(r'C:\Users\saivi\PycharmProjects\pythonProject12\data\face.h5')
