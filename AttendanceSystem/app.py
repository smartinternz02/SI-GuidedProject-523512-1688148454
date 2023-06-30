from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from keras.models import load_model
import datetime
import pandas as pd

app = Flask(__name__)


def create_folder(path, name):
    folder_path = os.path.join(path, name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def capture_images(folder_path, num_images, frame_interval):
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml.xml')

    image_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if frame_count % frame_interval == 0 and image_count < num_images:
                face_image = frame[y:y + h, x:x + w]

                image_path = os.path.join(folder_path, f"image{image_count + 1}.jpg")
                cv2.imwrite(image_path, face_image)
                image_count += 1

        cv2.putText(frame, f"Images taken: {image_count}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or image_count >= num_images:
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return image_count


def mark_attendance(label, attendance_df):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    if label not in attendance_df['Label'].values:
        attendance_df.loc[len(attendance_df)] = [label, date_string, time_string]


def recognize_person(frame, model, face_cascade, label_mapping, attendance_df):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_image = gray[y:y + h, x:x + w]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        face_image = cv2.resize(face_image, (224, 224))

        face_image = np.expand_dims(face_image, axis=0)
        face_image = face_image / 255.0

        predictions = model.predict(face_image)
        person_index = np.argmax(predictions)
        confidence = predictions[0][person_index]

        if confidence > 0.5:
            label = label_mapping[person_index]
            mark_attendance(label, attendance_df)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        id = request.form['id']
        # Process the registration data and perform necessary actions
        path = "D:/face"
        folder_path = create_folder(path, name)
        num_images = 10
        frame_interval = 10
        image_count = capture_images(folder_path, num_images, frame_interval)
        print(f"{image_count} images captured and stored in folder: {folder_path}")
        return "Registration complete"
    return render_template('register.html')


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    model = load_model("data/newmodel.h5")
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml.xml')

    dataset_path = 'D:/face'
    label_mapping = {}
    folders = os.listdir(dataset_path)
    for i, folder in enumerate(folders):
        label_mapping[i] = folder
    attendance_df = pd.DataFrame(columns=['Label', 'Date', 'Time'])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        result_frame = recognize_person(frame, model, face_cascade, label_mapping, attendance_df)

        cv2.imshow("Face Recognition", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    file_path = 'data/attendance.xlsx'

    attendance_df.drop_duplicates(subset=['Label'], inplace=True)  # Remove duplicate entries

    attendance_df.to_excel(file_path, index=False)

    print("Attendance saved to:", file_path)

    return 'Attendance marked'


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=2000)
