import dlib
import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

############### folder path where images are stored ##################
path = 'imgdir'

############### creating blank arrays for storing images #############
images = []
allaccess = []
myList = os.listdir(path)
print('Images Present :',myList)

########## adding all the images to the array 'images[]' #############
for cl in myList :
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    allaccess.append(os.path.splitext(cl)[0])
print('Name of people in images :',allaccess)

############### encoding all the images stored in images[] ###########
def FindEncodings(images):
    encodelist = []
    for img in images :
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodedListKnown = FindEncodings(images)
print('Encoding Done')
print('Number of people allowed to access',len(FindEncodings(images)))

################## taking input from the camera ######################
cap = cv2.VideoCapture(0)

while True :
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faceLoc = face_recognition.face_locations(imgRGB)
    encode = face_recognition.face_encodings(imgRGB,faceLoc)

####### comparing the camera encodings with stored image encodings #######
    for en,loc in zip(encode,faceLoc) :
        matches = face_recognition.compare_faces(encodedListKnown,en)
        faceDis = face_recognition.face_distance(encodedListKnown,en)
        #print(matches)
        matchindex = np.argmin(faceDis)
        #print(matchindex)
        if matches[matchindex] :
            name = allaccess[matchindex]
            print(name)
            for (y1, x2, y2, x1) in faceLoc :
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)