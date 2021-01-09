# Credit: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='models/model.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='models/labels.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
                    
args = parser.parse_args()

# Ruta al modelo
PATH_TO_MODEL_DIR = args.model

# Ruta a labelmap
PATH_TO_LABELS = args.labels

# Procentaje de deteccióon
MIN_CONF_THRESH = float(args.threshold)

resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
import time
print('Loading model...', end='')
start_time = time.time()

# Cargar modelo TFLite
interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
# Cargar labels
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

interpreter.allocate_tensors()

#Cargar detalles del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
dim =(100,100)
# Iniciar calculo de FPS
frame_rate_calc = 1
freq = cv2.getTickFrequency()
print('Running inference for PiCamera')
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

while True:
    # Start timer (for calculating frame rate)
    current_count = 0
    Cp = 0 #Conteo de pernos
    Ca = 0 #Conteo de arandelas  
    Ct = 0 #Conteo de tuercas
    
    areas = [0]*50 #Areas para clasificacion
    area=0 #Area maxima
    t1 = cv2.getTickCount()

    #Tomar captura del stream
    frame1 = videostream.read()

    # Redimensionar acorde al modelo
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalizar si es un modelo flotante (Modelo no cuantizado)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Realizar inferencia
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Capturar resultados Boxes=predicciones, classe=clase predicha, Score=proncentaje de confianza
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects


    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):       
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            #Clasificacion por tamaño
            
            pieza = frame[ymin:ymax,xmin:xmax]
            gray_frame = cv2.cvtColor(pieza, cv2.COLOR_BGR2GRAY)
            mean = int(np.mean(gray_frame)) + 5

            _, threshold = cv2.threshold(gray_frame, mean, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imshow('threshold', threshold)
            if (classes[i]==0):
                Cp+=1
            if (classes[i]==1):
                Ca+=1
            if (classes[i]==2):
                Ct+=1
            f = 0
            for cnt in contours:
                areas[f] = cv2.contourArea(cnt)
            area=max(areas)

            # Draw label

            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            label_ymax = max(ymax, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, 'Area: '+str(area), (xmin, label_ymax+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            current_count+=1

            
        
    # Draw framerate in corner of frame
    string = 'Pernos: %s; Arandelas: %d; Tuercas: %d' % (Cp,Ca,Ct)
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.putText (frame,'Total Detection Count : ' + str(current_count),(15,65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    
    #ConteoJR
    cv2.putText(frame, string, (15, 105), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)# Draw label text

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object Detector', frame)
#    if ((current_count)!=0):
#        cv2.imshow('Piezas', piezas_juntas)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
print("Done")