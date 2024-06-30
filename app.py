from flask import Flask, render_template, Response, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
from model import Model
import os
import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the model
def get_predictor_model():
    model = Model()
    return model

model = get_predictor_model()

current_label = None  # Global variable to store the current label

# Define labels to capture
capture_labels = {'violence in office', 'street violence', 'fight on a street'}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None
video_file_path = None  # No initial video file path
frames_after_detection = 30
post_detection_frames = 0
video_count = 1  # Counter for video files
camera_stopped = False  # Flag to indicate if the camera has stopped

# Function to classify an image
def classify_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = np.array(image)
    prediction = model.predict(image=image)
    label_text = prediction['label'].title()
    return label_text

def generate():
    global current_label, video_writer, video_file_path, post_detection_frames, video_count, camera_stopped

    cap = cv2.VideoCapture(0)  # Open the default webcam (index 0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while not camera_stopped:
        ret, frame = cap.read()
        if not ret:
            camera_stopped = True
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_label = classify_image(frame_rgb)  # Update the global label variable

        # Check if we need to start capturing or continue capturing additional frames
        if current_label.lower() in capture_labels or post_detection_frames > 0:
            if video_writer is None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
                fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                video_file_path = f'captured_video_{video_count}_{timestamp}.mp4'
                video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (width, height))
                print(f'Started capturing video for label: {current_label}')
                video_count += 1

            video_writer.write(frame)
            print(f'Capturing video for label: {current_label}')
            if current_label.lower() in capture_labels:
                post_detection_frames = frames_after_detection  # Reset post-detection frame counter
            else:
                post_detection_frames -= 1  # Decrement post-detection frame counter

        # Ensure the frame retains its color
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if post_detection_frames <= 0 and video_writer is not None:
            video_writer.release()
            print(f'Video writer released: {video_file_path}')
            video_writer = None  # Reset video_writer to None after release
            video_file_path = None  # Reset video_file_path after release

    cap.release()
    print("Camera stopped and released")

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/label')
def get_label():
    global current_label
    return jsonify(label=current_label)

@app.route('/captured_videos')
def get_captured_videos():
    videos = []
    for filename in os.listdir('.'):
        if filename.startswith('captured_video') and filename.endswith('.mp4'):
            videos.append(filename)
    return jsonify(videos=videos)

@app.route('/download/<filename>')
def download_video(filename):
    return send_file(filename, as_attachment=True)

@app.route('/delete/<filename>')
def delete_video(filename):
    os.remove(filename)
    return jsonify(message='Video deleted successfully')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    