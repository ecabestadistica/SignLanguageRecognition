# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:32:22 2019

@author: Elisa, Manuel
"""

import tensorflow as tf 
import h5py
import os
import cv2
import numpy as np
currentScriptPath = os.path.dirname(os.path.abspath(__file__))

############# General Config ################

camera_capture_rect = (50, 50, 200, 200)
camera_index = 0
main_window_name = "Video Feed"
example_window_name = "Example"

########### General Config ##############

# ask user to run numbers of letters model
while True:
    model_to_run = input("Run numbers or letters model? (N/L): ")
    if model_to_run.lower() == "n":
        break
    elif model_to_run.lower() == "l":
        break
    else:
        print("Please enter 'n' or 'l'")
        continue

if model_to_run.lower() == "n":
    ############# Model Config ################

    #model_path='./TrainedModels/numbers_model_100x100_70pc.keras'
    model_path='./TrainedModels/numbers_model_100x100_70pc.h5'
    image_side=100
    classes = ("0","1","2","3","4","5","6","7","8","9") # since our classes have just 1 character this can be also "0123456789" but we prefer to use a tuple
    example_images_path = currentScriptPath + '/Signos Numeros'

    ########### End Model Config ##############

elif model_to_run.lower() == "l":
    ############# Model Config ################

    model_path='./TrainedModels/letters_model_192x192_74.5pc.h5'
    image_side=192
    classes = ("A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S", "T", "U", "V", "W", "X", "Y") # since our classes have just 1 character this can be also "ABCDEFGHIKLMNOPQRSTUWXY" but we prefer to use a tuple
    example_images_path = currentScriptPath + '/Signos ASL'

    ########### End Model Config ##############


print(tf.__version__)
print(tf.config.list_physical_devices())

try:
    model=tf.keras.models.load_model(model_path)
except:
    ## print original exception
    import traceback
    print(traceback.format_exc())
    print("If you get error \"expected 2 variables, but received 0 variables during loading.\" try to change model path from .keras to .h5")
    exit()

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":

    # get list of example images
    list_of_example_images = os.listdir(example_images_path)

    # get the reference to the webcam
    try:
        camera = cv2.VideoCapture(camera_index)
    except:
        print("Error opening camera. Make sure you have a camera and try to change camera_index")
        exit()

    # get position from camera_capture_rect
    x1, y1, x2, y2 = camera_capture_rect

    firt_image_show = False
    last_example = 0
    last_feed_position = 0
    
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the image inside the camera_capture_rect
        captured_box = frame[x1:x2, y1:y2]

        # if captured_box shape is not the same of image_side, resize it
        if captured_box.shape[0] != image_side or captured_box.shape[1] != image_side:
            # resize the frame
            captured_box = cv2.resize(captured_box, (image_side, image_side), interpolation = cv2.INTER_AREA)

        # convert image from 0-255 to 0-1
        captured_box = captured_box / 255

        pred = model.predict(captured_box.reshape(-1, image_side, image_side, 3))

        # index of predicted class
        index = np.argmax(pred)

        # text to display
        predictionText = "Pred: " + classes[index]
        percent = round(pred[0][index] * 100, 2)
        predictionText += " " + str(percent) + "%"

         # color for our box and background of text
        mainColor = (95,0,189)
        
        cv2.rectangle(clone, (x1, y1), (x2, y2), mainColor, 2) # draw a box around the region of interest

        # fill background of the text
        scale = 0.8
        thinkness = 1
        separation = 6
        text_size, _ = cv2.getTextSize(predictionText, cv2.FONT_HERSHEY_SIMPLEX, scale, thinkness)
        text_width, text_height = text_size
        cv2.rectangle(clone, (x1, y1-text_height-separation-2), (x1 + text_width, y1-separation+2), mainColor, -1)
        cv2.putText(clone, predictionText, (x1, y1-separation), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thinkness)

        # display the frame with segmented hand
        cv2.imshow(main_window_name, clone)

        
        if cv2.waitKey(1) == ord(" ") or not firt_image_show:
            firt_image_show = True
            example_to_show = last_example
            while example_to_show == last_example:
                example_to_show = list_of_example_images[np.random.randint(len(list_of_example_images))]
            
            last_example = example_to_show
            
            # read the example image
            example_image = cv2.imread(example_images_path + '/' + example_to_show)
            example_image = cv2.resize(example_image, (150, 150), interpolation = cv2.INTER_AREA)
            
            # Using cv2.putText() method 
            example_image = cv2.putText(example_image, str(example_to_show[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA) 
   
            # Displaying the image 
            cv2.imshow(example_window_name, example_image)

        try:
            # move "Example" window to the right of "Video Feed" window
            video_feed_window_position = cv2.getWindowImageRect(main_window_name)
            if last_feed_position != video_feed_window_position:
                last_feed_position = video_feed_window_position
                cv2.moveWindow(example_window_name, video_feed_window_position[0] + video_feed_window_position[2] + 10, video_feed_window_position[1])
                # keep on front
                cv2.setWindowProperty(example_window_name, cv2.WND_PROP_TOPMOST, 1)
        except:
            pass

        # if the user pressed "q", then stop looping
        if cv2.waitKey(1)  == ord("q"):
            break

        # if Video Feed or Example window is closed, then stop looping
        if cv2.getWindowProperty(main_window_name, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(example_window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

# free up memory
camera.release()
cv2.destroyAllWindows()

