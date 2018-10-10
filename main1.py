#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response
from camera1 import VideoCamera
from wide_resnet import WideResNet
import os
from keras.utils.data_utils import get_file
from utils.datasets import get_labels
from keras.models import load_model
import cv2
import dlib
from drowsy import drowse
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
from models import yolo
# import playsound

import imutils
import time
import eval
from utils.general import format_predictions, find_class_by_name, is_url

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
import estimate_head_pose

global flag
global flag1

CASE_PATH = "/home/merly/Documents/gender/pretrained_models/haarcascade_frontalface_alt.xml"
WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
face_size = 64
depth=16
width=8
alpha=0.8

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/videos/", methods=['POST'])
def goToVideos():
    print('called route to videos')
    return render_template('index1.html');

@app.route("/videos1/", methods=['POST'])
def goToVideos1():
    print('called route to videos')
    return render_template('index2.html');
@app.route("/videos2/", methods=['POST'])
def goToVideos2():
    print('called route to videos')
    return render_template('index3.html');

@app.route("/videos3/", methods=['POST'])
def goToVideos3():
    print('called route to videos')
    return render_template('index4.html');


def gen(camera):
    model = None
    if model == None:
        model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5', WRN_WEIGHTS_PATH, cache_subdir=model_dir)
        print('**********************', fpath)
        fpath='/home/merly/Documents/project/pretrained_models/weights.18-4.06.hdf5'
        model.load_weights(fpath)
     
        emotion_model_path = '/home/merly/Documents/project/models/emotion_model.hdf5'
        emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        frame_window = 10
        emotion_offsets = (20, 40)

        # loading models
        face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        emotion_classifier = load_model(emotion_model_path)

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]

    # new_obj=VideoCamera()

    while True:
        frame = camera.get_frame(model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen1(camera):
    """Video streaming generator function."""
    detector=None
    if detector==None:

        shape_pred='shape_predictor_68_face_landmarks.dat'

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_pred)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        frame = camera.get_frame1(detector,predictor,lStart,lEnd,rStart,rEnd)
        # print (frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    

def gen2(camera):
    """Video streaming generator function."""
    model_cls=None
    flag=True
    if model_cls==None:
        
        model_cls = find_class_by_name('Yolo2Model', [yolo])
  
        

    while True:
      
        
        frame,flag = camera.get_frame2(model_cls,flag)
        # print (frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')   

def gen3(camera):
    """Video streaming generator function."""
    mark_detector=None
    flag1=True
    if mark_detector==None:
        mark_detector = MarkDetector()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    # height, width = sample_frame.shape[:2]
    # pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
        pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

    while True:
      
        
        frame ,flag1= camera.get_frame3(mark_detector, pose_stabilizers,flag1)
        # print (frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/videocam1')
def video_feed():

    global model
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videocam2')
def video_picamera1():
    print ('insede')
    """Video streaming route. Put this in the src attribute of an img tag."""
    #from mycodo.mycodo_flask.camera.camera_picamera import Camera
    return Response(gen1(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videocam3')
def video_feed_object():

    global model
    return Response(gen2(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videocam4')
def video_feed_headpose():

   
    return Response(gen3(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4002, debug=True)