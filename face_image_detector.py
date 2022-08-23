import cv2
from random import randrange

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("trains/haarcascade_frontalface_default.xml")
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# image to detect faces in
img = cv2.imread('images/face.jpg')

# covert image to gray scale
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
# (98, 30) top left corner coordinates, (86, 86) bottom right coordinates
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

# Draw rectangles around the faces
# # face_coordinates = [[98 30 86 86]]
(x, y, a, b) = face_coordinates[0]  # extract values out of the array coordinates for a single image
# (x,y) top coordinates, (x+a,y+b) bottom coordinates, (0,255,0) BGR green color, 2 rectangle thickness
cv2.rectangle(img, (x, y), (x + a, y + b), (0, 255, 0), 2)

cv2.imshow('Face Detector Application', img)
# closes window till a key is pressed
cv2.waitKey()
