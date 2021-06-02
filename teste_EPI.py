#Tensorflow só pode ser instalado em python 64bits, portanto utilize o Venv de Machine Learning
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
import io
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Load the model
model = tensorflow.keras.models.load_model('/home/pi/tm_test/L_tube_script/keras_model.h5')
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

time.sleep(0)

# Replace this with the path to your image
camera = PiCamera()
camera.resolution = (224,224)
rawCapture = PiRGBArray(camera, size=(224,224))
camera.start_preview()
for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):


#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center

#turn the image into a numpy array
    rawCapture.truncate()
    rawCapture.seek(0)
    img = np.copy(frame1.array)
    img = np.array(img,dtype=np.float32)
    img = np.expand_dims(img,axis=0)


# Normalize the image
    normalized_image_array = (img.astype(np.float32) / 127.0) - 1

# Load the image into the array
    data[0] = normalized_image_array

# run the inference
    prediction = model.predict(data)
    print(prediction)

    if prediction[0][0] >= 0.7:
        print('O usuário está com EPI')
    else:
        print('O usuário está sem EPI')
    if cv2.waitKey(1)==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()