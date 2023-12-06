
import keras              
from keras.models import load_model
import cv2
import numpy as np
import os
import gradio as gr



model = load_model(r'model/model.h5')


img =  r'images/1679394454875.jpg'


def predict_input_image(img):
    
    #(1)Load the image
    # read the image using OpenCV
    image = cv2.imread(img) 
    
    #(2)Preprocess the image
    # resize the image to the required size
    image = cv2.resize(image, (224, 224))  

    # the class names for the labels
    class_names = ['fire_images', 'non_fire_images']  


    img_4d = image.reshape(-1, 224, 224, 3)  


    prediction = model.predict(img_4d)[0]   

   # If the prediction value is greater than 0.5, it is considered as fire
    if prediction > 0.5:   
        # probability of fire and non-fire
        pred = [1-prediction, prediction]  
    else:
         # probability of non-fire and fire
         pred = [1-prediction, prediction]  
    

    confidences = {class_names[i]: float(pred[i]) for i in range(2)}  

    # return the dictionary of class names and their probabilities
    return confidences   

# that's how we call the function
result = predict_input_image(img)

# print the result
print(result)


def predict_input_image_gr(img):
   
    # the class names for the labels
    class_names = ['fire_images', 'non_fire_images']   

    img_4d = img.reshape(-1, 224, 224, 3)


    prediction = model.predict(img_4d)[0]

    # If the prediction value is greater than 0.5, it is considered as fire
    if prediction > 0.5:
        # Probability of fire and non-fire
        pred = [1-prediction, prediction]
    else:
        # Probability of non-fire and fire
        pred = [1-prediction, prediction]

    confidences = {class_names[i]: float(pred[i]) for i in range(2)}

    # Return the dictionary of class names and their probabilities
    return confidences



# Define the input for the Gradio interface as an image with a shape of 224x224
image = gr.inputs.Image(shape=(224, 224))

# Define the output for the Gradio interface as a label with one class
label = gr.outputs.Label(num_top_classes=1)


gr.Interface(fn=predict_input_image_gr, inputs=image, outputs=label, interpretation='default').launch(debug='True', share='True')
