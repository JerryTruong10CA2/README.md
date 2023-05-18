import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime

# TẠO ĐƯỜNG DẪN ĐẾN DATASET CỦA ẢNH
path = "pics2"
images = []
Names = []
myList = os.listdir(path)
for image in myList:
    curImg = cv2.imread(f"{path}/{image}")
    images.append(curImg)
    Names.append(os.path.splitext(image)[0])

# MÃ HÓA ẢNH
def Encoding(images):
    encode = []
    for img in images:
        encoding = face_recognition.face_encodings(img)[0]
        encode.append(encoding)
    return encode
encoded_img = Encoding(images)
# GHI CHÚ LẠI THỜI GIAN NHẬN BIẾT KHUÔN MẶT THÀNH CÔNG
def check_in(name):
    with open('check_in.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList and name !='Unknown':
            now = datetime.now()
            f.writelines(f'\n{name},{now}')

cam = cv2.VideoCapture(0) #KHỞI ĐỘNG CAMERA

while True:
    # NHẬN DIỆN KHUÔN MẶT
    ret, frame = cam.read()
    frame_resize = cv2.resize(frame, (0, 0), None, fx=1, fy=1)

    face_in_Frame = face_recognition.face_locations(frame_resize)
    encode_in_Frame = face_recognition.face_encodings(frame_resize)

    for encodeFace, faceLoc in zip(encode_in_Frame, face_in_Frame):
        matches = face_recognition.compare_faces(encoded_img, encodeFace)
        faceDis = face_recognition.face_distance(encoded_img, encodeFace)
        matchIndex = np.argmin(faceDis) # LẤY INDEX CỦA TẤM ẢNH & TÊN TƯƠNG ỨNG CÓ ĐỘ KHÁC BIỆT NHỎ NHẤT SO VỚI FRAME

        if faceDis[matchIndex] < 0.5:
            name = Names[matchIndex].upper()
            check_in(name)
        else:
            name = "Unknown"

        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # VẼ KHUNG XUNG QUANH KHUÔN MẶT NHẬN DIỆN ĐƯỢC TRONG FRAME
        if name != "Unknown":
            cv2.putText(frame, "Welcome " + name + ' ' + str(round((1 - faceDis[matchIndex]) * 100, 0)), (x1, y2 + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 255, 255), 2) # VIẾT RA TÊN + ĐỘ CHÍNH XÁC
        else:
            cv2.putText(frame, name, (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('face_recognition', frame)
    if cv2.waitKey(1) == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()


# ĐĂNG KÝ KHUÔN MẶT NẾU NHƯ CHƯA NHẬN BIẾT ĐƯỢC
if name == 'Unknown':
    decision = input('Type the password if you want to sign up (if not then skip this step): ')
    if decision == '25102007':
        face_detector = cv2.CascadeClassifier('Learning-Samples/haarcascade_frontalface_alt.xml')
        name_sign= input("name: ")
        while True:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            face = face_detector.detectMultiScale(frame, 1.3, 5)
            for x, y, w, h in face:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.imshow('Face_Dectector', frame)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.imwrite('pics2/' + name_sign + '.jpg', frame, (100, 100))
        cam.release()
        cv2.destroyAllWindows()

