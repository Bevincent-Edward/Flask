from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import os

app = Flask(__name__)

people = ['aakash', 'bevin']
DIR = r'D:\VMS'
haar_cascade = cv.CascadeClassifier('haar.xml')
features = []
labels = []
face_recognizer = cv.face.LBPHFaceRecognizer_create()

def create_train():
    global features, labels, face_recognizer
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer.train(features, labels)
    face_recognizer.save('face_trained.yml')
    np.save('features.npy', features)
    np.save('labels.npy', labels)


def recognize_face(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)

        if confidence > 100:
            return "Imposter"
        else:
            return people[label]


@app.route('/')
def index():
    return render_template('vmsfront.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    img = cv.imdecode(np.fromstring(file.read(), np.uint8), cv.IMREAD_COLOR)

    recognized_person = recognize_face(img)

    return jsonify({'recognized_person': recognized_person})


if __name__ == '__main__':
    create_train()  # Call create_train function before running the app
    app.run(debug=True)