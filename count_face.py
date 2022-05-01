
import cv2
import os

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:

    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(

        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    loc1 = (30, 50)
    loc2 = (550, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 100, 255)
    thickness = 2
    if faces == ():
      num_face = 0
      frames = cv2.putText(frames, "No Face Detected", loc1, font,
                           fontScale, color, thickness, cv2.LINE_AA)
    else:
      num_face = faces.shape[0]

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frames = cv2.putText(frames, "Total Number of Face Detected", loc1, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        frames = cv2.putText(frames, str(num_face), loc2, font,
                             fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
