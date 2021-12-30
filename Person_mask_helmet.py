# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pandas as pd
import tensorflow_hub as hub

# Person Detection
os.environ['TFHUB_CACHE_DIR'] = '/Users/neerajtadur/Downloads/Face-Mask-Detection-master'

# Person Detection
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

#Function to detect face and predict if the person is wearing a mask or not
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

##make change here
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])
helmetNet = load_model('Helmet_detector.model')


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

width = 512
height = 512

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	#frame = imutils.resize(frame, width=400)

	# Person Detection Module
	frame = cv2.resize(frame, (width , height ))

	rgb_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)

	#Add dims to rgb_tensor
	rgb_tensor = tf.expand_dims(rgb_tensor, 0)
	
	boxes, scores, classes, num_detections = detector(rgb_tensor)
	
	pred_labels = classes.numpy().astype('int')[0]
	
	pred_labels = [labels[i] for i in pred_labels]
	pred_boxes = boxes.numpy()[0].astype('int')
	pred_scores = scores.numpy()[0]
	#loop throughout the faces detected and place a box around it

	for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
		if score < 0.5:
			continue
			
		score_txt = f'{100 * round(score,0)}'
		frame = cv2.rectangle(frame,(xmin, ymax),(xmax, ymin),(0,255,0),1)      
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
		cv2.putText(frame,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)


	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	# detect helmet in the frame and determine if they are wearing a
	# helmet or not
	(locs1, preds1) = detect_and_predict_mask(frame, faceNet, helmetNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box1, pred1) in zip(locs1, preds1):
		# unpack the bounding box and predictions
		(startX1, startY1, endX1, endY1) = box1
		(helmet, withouthelmet) = pred1

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label1 = "Helmet" if helmet > withouthelmet else "No Helmet"
		color1 = (0, 255, 0) if label1 == "Helmet" else (0, 0, 255)
			
		# include the probability in the label
		label1 = "{}: {:.2f}%".format(label1, max(helmet, withouthelmet) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label1, (startX1, startY1 - 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color1, 2)
		cv2.rectangle(frame, (startX1, startY1), (endX1, endY1), color1, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()