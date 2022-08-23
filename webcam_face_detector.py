import cv2
from random import randrange

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("trains/haarcascade_frontalface_default.xml")

# capture video from webcam, 0 default-camera
webcam = cv2.VideoCapture(0)

# Iterate over the frame
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()

    # convert image frames to grayscale
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    # (98, 30) top left corner coordinates, (86, 86) bottom right coordinates
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

    # Draw rectangles around the faces
    # # face_coordinates = [[98 30 86 86]]
    # (x, y, a, b) = face_coordinates[0]  # extract values out of the array coordinates for a single image
    # # (x,y) top coordinates, (x+a,y+b) bottom coordinates, (0,255,0) BGR green color, 2 rectangle thickness
    # cv2.rectangle(img, (x, y), (x + a, y + b), (0, 255, 0), 2)
    for (x, y, a, b) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + a, y + b), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Face Detector Application', frame)
    # waitKey with 1 will load frames at every 1 milliseconds
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        # Release the VideoCapture object
        webcam.release()
        break
