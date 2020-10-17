# import cv2
# import numpy as np
# import face_recognition
# import os
#
# path='ImagesBasic'
# classNames=[]
# images=[]
# myList=os.listdir(path)
# print(myList)
#
# for cls in myList:
#     currentImage=cv2.imread(f'{path}/{cls}')
#     images.append(currentImage)
#     classNames.append(os.path.splitext(cls)[0])
# print(classNames)
#
# def findEncodings(images):
#     encodeList=[]
#     for img in images:
#         img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         Enc = face_recognition.face_encodings(img)[0]
#         encodeList.append(Enc)
#     return encodeList
# encodeListKnown=findEncodings(images)
# print('Encoding Complete')
#
# cap=cv2.VideoCapture(0)
# while True:
#     success,img=cap.read()
#     imgS=cv2.resize(img,(0,0),None,0.25,0.25)
#     imgS= cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     facesCurrentFrame = face_recognition.face_locations(imgS)
#     encCurrentFrame = face_recognition.face_encodings(imgS,facesCurrentFrame)
#
#     for encodeFace,faceLoc in zip(encodeListKnown,facesCurrentFrame):
#         matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
#         print(faceDist)
#         matchIndex=np.argmin(faceDist)
#         if(matches[matchIndex]):
#             name=classNames[matchIndex].upper()
#             print(name)
#
#


# imgSid=face_recognition.load_image_file('ImagesBasic/Sid.JPG')
# imgSid=cv2.cvtColor(imgSid,cv2.COLOR_BGR2RGB)
# imgTest=face_recognition.load_image_file('ImagesBasic/Sid_test.JPG')
# imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'ImagesBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def MarkAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if(name not in nameList):
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')





# for line in myDataList:
#     entry = line.split(',')
# nameList.append(entry[0])
# if name not in nameList:
#     now = datetime.now()
# dtString = now.strftime('%H:%M:%S')
# f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        # matchIndex = np.argmin(faceDist)
        if (matches[matchIndex]):
            name = classNames[matchIndex].upper()
            # print(name)

            y1,x2,y2,x1=faceLoc
            x1,x2,y1,y2=x1*4,x2*4,y1*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,244))
            MarkAttendance(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
