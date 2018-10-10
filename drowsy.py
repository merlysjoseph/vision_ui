from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
# import playsound
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

shape_pred='shape_predictor_68_face_landmarks.dat'

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES1 = 3

# # initialize the frame counters and the total number of blinks
COUNTER1 = 0
TOTAL1 = 0
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shape_pred)

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# vs = cv2.VideoCapture(-1)
# time.sleep(1.0)

# loop over frames from the video stream
def drowse(frame,detector,predictor,lStart,lEnd,rStart,rEnd):
	global EYE_AR_THRESH 
	global EYE_AR_CONSEC_FRAMES 
	global COUNTER 
	global ALARM_ON 

	global EYE_AR_THRESH 
	global EYE_AR_CONSEC_FRAMES1 

	# # initialize the frame counters and the total number of blinks
	global COUNTER1 
	global TOTAL1 
	
	if 1:
		# frame = imutils.resize(image, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# gray=frame

		# detect faces in the grayscale frame
		rects = detector(gray, 1)
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			if ear < EYE_AR_THRESH:
				COUNTER += 1
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					# if the alarm is not on, turn it on
					if not ALARM_ON:
						ALARM_ON = True
					# draw an alarm on the frame
					cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# otherwise, the eye aspect ratio is not below the blink
			# threshold, so reset the counter and alarm
			else:
				COUNTER = 0
				ALARM_ON = False

			if ear < EYE_AR_THRESH:
				COUNTER1 += 1

			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else :
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER1>= EYE_AR_CONSEC_FRAMES1:
					TOTAL1 += 1

				# reset the eye frame counter
				COUNTER1 = 0
				# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL1), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	return frame


# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF

# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()




