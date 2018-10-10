import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
alpha =0.8
def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(emotion_probability,coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=0.5, thickness=3,
                                                ):
    x, y = coordinates[:2]
    # z=coordinates[2:3]
    # cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                # cv2.FONT_HERSHEY_SIMPLEX,
                # font_scale, color, thickness, cv2.LINE_AA)
    # print (text)
    if text=='happy'or text == 'neutral':
        # print ('yes')
        string =text+ str(emotion_probability)
    else:
        # print ('noooooooooooooooooo')
        string =text+ str(emotion_probability)
    # overlay=image_array.copy()
    # cv2.rectangle(overlay, (x, y), (x + z, y-40), (255, 255, 255),-1)
    # cv2.addWeighted(overlay, alpha , image_array, 1 - alpha,0,image_array)
    cv2.putText(image_array, string, (x , y-20 ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255,255,255), thickness, lineType=2)

        

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors

