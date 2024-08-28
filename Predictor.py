from keras.models import load_model
import keras.utils as image
import numpy as np
import os

dirname = os.path.dirname(__file__)

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


prediction=vgg19.predict(img, batch_size=10)

maxAccuracyFake = 0
maxAccuracyReal = 0

print(prediction)

print((str(round(100 * prediction[0][0], 2)) + " percent chance of being a deepfake") if prediction[0][0] > prediction[0][1] else (str(str(round(100 * prediction[0][1], 2))) + " percent chance of being real!"))

