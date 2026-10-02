from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load model
model=YOLO("yolov8s.pt")
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Webcam Detection
def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(frame, conf=0.6)

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Image Upload Detection
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['image']

    if file:
        import os

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filepath)

        results = model(filepath, conf=0.6)

        annotated_frame = results[0].plot()

        result_path = os.path.join(app.config['UPLOAD_FOLDER'], "result.jpg")

        cv2.imwrite(result_path, annotated_frame)

        return render_template('index.html',
                               uploaded=True,
                               result_image=result_path)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
