import numpy as np
import cv2
import os
import keras
from keras.models import load_model
import matplotlib.pyplot as plt


# Define the path of the input image and load the pre-trained model
img = r'images/fire.90.png'
model = load_model(r'model/model.h5')

predi = 52
# Define the path of the input image and load the pre-trained model
def predict_input_image(img):
    class_names = ['fire_images', 'non_fire_images']

    # Reshape the input image into a 4D array
    img_4d = img.reshape(-1, 224, 224, 3)

    # Use the loaded model to predict the class of the input image
    prediction = model.predict(img_4d)[0]

   # If the prediction value is greater than 0.5, it is considered as fire
    if prediction > 0.5:   
        # probability of fire and non-fire
        pred = [1-prediction, prediction]  
    else:
         # probability of non-fire and fire
         pred = [1-prediction, prediction]  

    # Create a dictionary containing the confidence scores for each class
    confidences = {class_names[i]: float(pred[i]) for i in range(2)}
    print(confidences)
    return confidences

# Open a video 
#cam = cv2.VideoCapture("C:/Users/tarek/Desktop/W.D/EgyBest .The.Walking.Dead.S04E16.BluRay.360p.x264.mp4")

# Open a video by camera
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print('Camera not found')
    exit()

cam.set(3,224)
cam.set(4, 224)

width =cam.get(3)
height =cam.get(4)

print(width, height)

# Loop through each frame of the video
while True:

    # Loop through each frame of the video
    ret, frame = cam.read()

     # If there are no more frames, exit the loop
    if not ret:
        print('Can not receive frame (stream end?). Exiting ...')
        break

    # Resize the frame to a desired size for model function
    resized_frame = cv2.resize(frame, (224, 224))


    
    # Use the loaded model to predict the class of the resized frame
    pred = predict_input_image(resized_frame)

    # Determine the class label with the highest confidence score and create a result string
    if pred['fire_images'] > pred['non_fire_images'] and round(pred['fire_images']*100, 2) > predi :
        res = f"Fire {round(pred['fire_images']*100, 2)} %"
    else:
       # res = f"No Fire {round(pred['non_fire_images']*100, 2)} %"
        res = "No Fire"

    # Resize the original frame and overlay the result string on top of it   
    frame = cv2.resize(frame, (640, 480))

    # Put text and predection of detected fire on picture
    resized_frame = cv2.putText(frame, str(res), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', resized_frame)


    # Create a mask for the image by filtering it based on a certain color range
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50], dtype='uint8')
    upper_bound = np.array([35, 255, 255], dtype='uint8')
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
     
    # Calculate the normalized histogram of the filtered image and display it
    fire_hist = cv2.calcHist([masked_frame], [0], mask, [256], [0, 256])
    
    # This line normalizes the fire_hist using cv2.normalize() function. 
    # The resulting values are scaled between 0 and 255, 
    # which are the minimum and maximum values of an 8-bit image.
    fire_hist_norm = cv2.normalize(fire_hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #cv2.imshow('',masked_frame)


    # This block of code plots the normalized histogram
    # using Matplotlib's plot() function.
    # It also sets the x and y limits of the plot
    # and adds a title and labels for the axes.

    if pred['fire_images'] > pred['non_fire_images'] and round(pred['fire_images']*100, 2) > predi :
      plt.plot(fire_hist_norm)
    else:
         plt.plot(0)
    plt.xlim([0, 256])
    plt.ylim([0, 1000])
    plt.title("Fire Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show(block=False)

    # This line pauses the plot for 0.1 seconds to allow time 
    # for the plot to be displayed. 
    plt.pause(0.1)

    # Clears the plot for the next iteration.
    plt.clf()


    # This line waits for the 'q' key to be pressed to break the while loop
    # and exit the program. The waitKey() function waits 
    # for a specified delay (in milliseconds) 
    # for a key event to occur. If the pressed key's ASCII
    # value matches that of 'q', then the loop is broken.
    if cv2.waitKey(1) == ord('q'):
        break









# image = gr.inputs.Image(shape=(224, 224))
# label = gr.outputs.Label(num_top_classes=1)
#
#
# gr.Interface(fn=predict_input_image,
# inputs=image, outputs=label,interpretation='default').launch(debug='True', share='True')