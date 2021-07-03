import PIL 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.applications.resnet50 import ResNet50 
from keras.applications import resnet50


class ResNET:
    def __init__(self, ):
         
        # create resnet model 
        self.resnet_model = resnet50.ResNet50(weights = 'imagenet')

    def run_object_prediction(self, filename):
        self.original = load_img(filename, target_size = (224, 224)) 

        #convert the PIL image to a numpy array 
        numpy_image = img_to_array(self.original) 
        # Convert the image / images into batch format 
        self.image_batch = np.expand_dims(numpy_image, axis = 0) 
        # prepare the image for the resnet50 model 
        processed_image = resnet50.preprocess_input(self.image_batch.copy())
        # get the predicted probabilities for each class 
        predictions = self.resnet_model.predict(processed_image)
        # convert the probabilities to class labels 
        label = decode_predictions(predictions)
        return label