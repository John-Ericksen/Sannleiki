from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ConvNeXtLarge

num_classes = 2

image_resize = 224

batch_size_training = 60
batch_size_validation = 60
face_data_generator = ImageDataGenerator(
)

face_train_generator = face_data_generator.flow_from_directory(
    "data/face_data/train",
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

face_validation_generator = face_data_generator.flow_from_directory(
    "data/face_data/valid",
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')


vgg16FaceModel = Sequential()
vgg16FaceModel.add(ConvNeXtLarge(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
vgg16FaceModel.add(Dense(num_classes, activation='softmax'))
vgg16FaceModel.layers[0].trainable = False
vgg16FaceModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch_training = len(face_train_generator)
steps_per_epoch_validation = len(face_validation_generator)
num_epochs = 5

fit_history = vgg16FaceModel.fit(
    face_train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=face_validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)

vgg16FaceModel.save('face_classifier_ConvNeXtLarge.h5')