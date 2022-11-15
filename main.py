import cv2
import face_recognition
from flask import Flask, render_template
from flask import jsonify
from flask import flash, request
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
import numpy as np



import time
import os
import pickle
from datetime import datetime
import subprocess


classNames = list()
app = Flask(__name__)


@app.route('/')
def load_homepage():
    return render_template('index.html')


@app.route('/capture', methods=['GET'])
def recognize_face():
    images_list = []
    paths = 'images'

    img_list = os.listdir(paths)
    for cl in img_list:
        print("In Encode")
        img = cv2.imread(f"{paths}/{cl}")
        images_list.append(img)
        classNames.append(os.path.splitext(cl)[0])

    encoded_face_train = find_encodings(images_list)

    ENCODE_TRAIN = encoded_face_train
    encoded_face_train = ENCODE_TRAIN

    # take pictures from webcam
    cap  = cv2.VideoCapture(0)

    # setting timeout to 20 seconds
    timeout = time.time() + 20
    flag = 0
    person_name = ""
    while True:
        success, img = cap.read()

        #initially 
        if flag == 0 :
            time.sleep(1)
            flag = 1
        if time.time() > timeout:
            cap.release()
            cv2.destroyAllWindows()
            break
            quit()
        try :

            cv2.imshow('Webcam', img)
            imgS = cv2.resize(img, (0,0), None, 1, 1)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faces_in_frame = face_recognition.face_locations(imgS)
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

            for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
                matches = face_recognition.compare_faces(encoded_face_train, encode_face)
                faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
                matchIndex = np.argmin(faceDist)
                print(matchIndex)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper().lower()
                    y1, x2, y2, x1 = faceloc
                    
                    y1, x2, y2, x1 = y1, x2, y2, x1
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    person_name = name
                    time.sleep(2)
                    markAttendence(name)
                    cap.release()


            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except :
            cap.release()
            cv2.destroyAllWindows()
            break
            quit()
    


    return "Attendance marked Mr." + person_name

def get_wifi_name() :
    try :
        subprocess_result = subprocess.Popen('iwgetid',shell=True,stdout=subprocess.PIPE)
        subprocess_output = subprocess_result.communicate()[0],subprocess_result.returncode
        network_name = subprocess_output[0].decode('utf-8')
    except :
        pass

    try :
        process = subprocess.Popen(['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport','-I'], stdout=subprocess.PIPE)
        out, err = process.communicate()
        process.wait()

        wifi_info = {}
        for line in out.decode("utf-8").split("\n"):
            if ": " in line:
                key, val = line.split(": ")
                key = key.replace(" ", "")
                val = val.strip()

                wifi_info[key] = val
        network_name = wifi_info["SSID"]
    except :
        pass

    # if network_name

def find_encodings(images):
    encodeList = list()
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_face = face_recognition.face_encodings(img)[0]
        print(f"==== {encode_face}== ")
        encodeList.append(encode_face)
    return encodeList


def markAttendence(name):

    with open('Attendance.csv', 'r+') as f:
        datas = f.readlines()
        name_list = []
        for l in datas:
            entry = l.split(',')
            name_list.append(entry[0])

        if name not in name_list:
            time = time.strftime('%I:%M:%S:%p')
            date = time.strftime('%d-%B-%Y')
            f.write(f"{name}, {time}, {date}\n")
            f.close()


# main driver function
if __name__ == '__main__':
    app.run()