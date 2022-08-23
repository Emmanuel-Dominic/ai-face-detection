import cv2
from random import randrange

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("trains/haarcascade_frontalface_default.xml")
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# image to detect faces in
img = cv2.imread('images/faces.png')

# covert image to gray scale
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
# (98, 30) top left corner coordinates, (86, 86) bottom right coordinates
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

# Draw rectangles around the faces
for (x, y, a, b) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + a, y + b), (randrange(256), randrange(256), randrange(256)), 2)

cv2.imshow('Face Detector Application', img)
# closes window till a key is pressed
cv2.waitKey()
