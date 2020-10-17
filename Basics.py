import cv2
import numpy as np
import face_recognition

imgSid=face_recognition.load_image_file('ImagesBasic/Sid.JPG')
imgSid=cv2.cvtColor(imgSid,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('ImagesBasic/Sid_test.JPG')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgSid)[0]
faceEnc=face_recognition.face_encodings(imgSid)[0]
cv2.rectangle(imgSid,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(223,23,23),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
faceEncTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(223,23,23),2)

results=face_recognition.compare_faces([faceEnc],faceEncTest)
faceDis=face_recognition.face_distance([faceEnc],faceEncTest)
print (results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_ITALIC,1,(0,0,0),2)


cv2.imshow('Sid',imgSid)
cv2.imshow("Test",imgTest)
cv2.waitKey(0)