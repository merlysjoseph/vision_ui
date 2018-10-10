import cv2
from new1 import original_model   
from drowsy import drowse   
from eval import evaluate
from estimate_head_pose import pose
from pose_estimator import PoseEstimator
from estimate_head_pose import pose




global flag
global flag1
height=width=0
model1=None
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture('/home/user2/ui_vision_projects/projectmerly/video.mp4')
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels):

        success, image = self.video.read()
        # print(image)

        img=original_model(success,image,model,emotion_classifier,emotion_target_size,face_cascade,frame_window,emotion_offsets,emotion_labels)
        # cv2.imshow('Img',img)
        # cv2.waitKey(1)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
        # return img
    def get_frame1(self,detector,predictor,lStart,lEnd,rStart,rEnd):

        success, image = self.video.read()
        # print (image)
        img2=drowse(image,detector,predictor,lStart,lEnd,rStart,rEnd)
        ret, jpeg = cv2.imencode('.jpg', img2)
        return jpeg.tobytes()
    
    def get_frame2(self,model_cls,flag):
        global height,width,model1
        success, image = self.video.read()
        # # print (image)
        if flag==True:
            height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
            # h,w,_=image.shape
        
            model = model_cls(input_shape=(height,width, 3))
            model.init()
            model1=model
        model=model1
        img3=evaluate(image,model)
        flag= False
        ret, jpeg = cv2.imencode('.jpg', img3)
        return jpeg.tobytes(),flag

    def get_frame3(self,mark_detector, pose_stabilizers,flag1):
        global height, width
        success, image = self.video.read()
        print(image)
        if flag1==True:
            height, width = image.shape[:2]
            # height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)

            flag1= False
        pose_estimator = PoseEstimator(img_size=(height, width))
        img4=pose(image,mark_detector,pose_estimator,pose_stabilizers)
        ret, jpeg = cv2.imencode('.jpg', img4)
        return jpeg.tobytes(),flag1

    

#
# if __name__=='__main__':
#     new_obj=VideoCamera()
#     while True:
#       new_obj.get_frame()
