import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
# import dlib
from imutils import face_utils
import os
import uuid

from config import *
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

face_size = 64
depth = 16
width = 8
alpha = 0.8
# model=None
if 0:
    CASE_PATH = "/home/merly/Documents/gender/pretrained_models/haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
    face_size = 64
    depth = 16
    width = 8
    alpha = 0.8
    model = WideResNet(face_size, depth=depth, k=width)()
    model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
    fpath = get_file('weights.18-4.06.hdf5',WRN_WEIGHTS_PATH,cache_subdir=model_dir)
    print('**********************',fpath)
    model.load_weights(fpath)

    emotion_model_path = 'models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []



def draw_label(image, point, label_temp, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.5, thickness=1):
        
        # size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        # print("#####",point)
        x, y , w, h= point
        pt=(x,y)
        # size = cv2.getTextSize(label_temp, font, font_scale, thickness)
        # print("$##$",size)
        overlay = image.copy()
        
        cv2.rectangle(overlay, (x, y), (x + w, y-40), (128,128,128),-1)
        cv2.addWeighted(overlay, alpha , image, 1 - alpha,0,image)
        # cv2.imshow("out",image)

        cv2.rectangle(image, (x, y), (x + w, y+h), (255, 0, 0))
        cv2.putText(image, label_temp, pt, font, font_scale, (255, 255, 255), thickness,lineType=2)

def crop_face( imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
def detect_face(frame,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels):
    # global model
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray_image,scaleFactor=1.2,minNeighbors=10,minSize=(face_size, face_size))
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,minSize=(face_size, face_size), flags=cv2.CASCADE_SCALE_IMAGE)
    face_imgs = np.empty((len(faces), face_size, face_size, 3))
    label=""
    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, margin=40, size=face_size)
        (x, y, w, h) = cropped
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_imgs[i,:,:,:] = face_img

        if len(face_imgs) > 0:
# predict ages and genders of the detected faces
            results = model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            label = "age={}, gender={}".format(int(predicted_ages[i]),
                                                "F" if predicted_genders[i][0] > 0.5 else "M")
    return (faces,label)


def detect_emotion(frame,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        # gray_face = gray_image[y1:y2, x1:x2]
        # gray_c = cv2.cvtColor(gray_image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray_c = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_c, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        # print(gray_face)
        emotion_prediction = emotion_classifier.predict(gray_face)
        
        
        emotion_probability =round( np.max(emotion_prediction),2)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        # print(emotion_text)
        # print(emotion_window)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # if emotion_text == 'angry':
        #     color = emotion_probability * np.asarray((255, 0, 0))
        # elif emotion_text == 'sad':
        #     color = emotion_probability * np.asarray((0, 0, 255))
        # elif emotion_text == 'happy':
        #     # print (emotion_probability)
        #     color = emotion_probability * np.asarray((255, 255, 0))
        # elif emotion_text == 'surprise':
        #     color = emotion_probability * np.asarray((0, 255, 255))
        # else:
        #     color = emotion_probability * np.asarray((0, 255, 0))

        # color = color.astype(int)
        # color = color.tolist()
        color=(255,255,255)
        draw_text(emotion_probability,face_coordinates, frame, emotion_mode,color, 0, -20, 0.5, 1)

        # return frame

# face_cascade = cv2.CascadeClassifier(CASE_PATH)
# # cap = cv2.VideoCapture('/home/merly/Documents/gender/genderr.mp4')
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out_corner = cv2.VideoWriter('face_emo.avi',fourcc, 20.0, (640, 480))
def original_model(ret,frame,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels):

    # Capture frame-by-frame
    # ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if 1:


        face_1,label_new=detect_face(frame,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels)
        # print(label_new)
        # print(gray)

        # print("###",prob)
        for i, face in enumerate(face_1):


            draw_label(frame, (face[0],face[1],face[2],face[3]), label_new)
            # draw_text(prob,(face[0],face[1],face[2],face[3]), frame, mode_list,colorr, 0, -45, 1, 1)
            # bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detect_emotion(frame,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels)
        # cv2.imshow('Keras Faces', frame)
        # out_corner.write(frame)
    # else:
    #     break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    return frame

            
                
        # When everything is done, release the capture
        #video_capture.release()
    # cv2.destroyAllWindows()
    
                                


