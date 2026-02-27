import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img,img_to_array

IMG_SIZE = 150 
model = load_model("image_classifier_model.h5")


def predict_image(img_path):
    print("Loading:",img_path)
    img= load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        print("Prediction: Dog 🐶")
    else:
        print("Prediction: Cat 🐱")

    print("Confidence:", float(prediction[0][0]))


predict_image("dataset/validation/cats/cat1.jpg")
predict_image("dataset/validation/cats/cat2.jpg")
predict_image("dataset/validation/dogs/dog1.jpg")
predict_image("dataset/validation/dogs/dog2.jpg")
