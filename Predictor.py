from keras.models import load_model
import keras.utils as image
import numpy as np
import os

dirname = os.path.dirname(__file__)

efficientNetV2B3 = load_model(os.path.join(dirname, "models/face_classifier_EfficientNetV2B3.h5"))
efficientNetV2L = load_model(os.path.join(dirname, "models/face_classifier_EfficientNetV2L.h5"))
mobilev3Large = load_model(os.path.join(dirname, "models/face_classifier_mobilev3_large.h5"))
resnet50 = load_model(os.path.join(dirname, "models/face_classifier_resnet50.h5"))
vgg19 = load_model(os.path.join(dirname, "models/face_classifier_vgg19.h5"))

num_classes = 2
image_resize = 224
batch_size_training = 60
batch_size_validation = 60

def load_image(img_path):

    img = image.load_img(img_path, target_size=(image_resize, image_resize))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def load_user_image():
    img_path = input("Enter the path of the image you want to evaluate: ")
    img = load_image(img_path)
    if(img.shape):
        return img
    else:
        print("The path you entered is not a valid image. Please try again.")
        load_image()

img = load_user_image()

predictions = []
predictions.append(efficientNetV2B3.predict(img, batch_size=10))
predictions.append(efficientNetV2L.predict(img, batch_size=10))
predictions.append(mobilev3Large.predict(img, batch_size=10))
predictions.append(resnet50.predict(img, batch_size=10))
predictions.append(vgg19.predict(img, batch_size=10))

realPredictions = 0
fakePredictions = 0
maxAccuracyFake = 0
maxAccuracyReal = 0

index = 0
for prediction in predictions:
    if(prediction[0].argmax(axis=-1) == 1):
        fakePredictions +=1
        print ("Model " + str(index) + ": " + str(prediction[0][1]) + "\% chance of a deep fake")
        maxAccuracyFake = maxAccuracyFake if maxAccuracyFake > prediction[0][1] else prediction[0][1]
        
    else: 
        realPredictions +=1
        print ("Model " + str(index) + ": " + str(prediction[0][1]) + "\% chance of being real!")
        maxAccuracyReal = maxAccuracyReal if maxAccuracyReal> prediction[0][1] else prediction[0][1]


print((str(round(100 * maxAccuracyFake, 2)) + " percent chance of being a deepfake") if fakePredictions > realPredictions else (str(str(round(100 * maxAccuracyReal, 2))) + " percent chance of being real!"))

