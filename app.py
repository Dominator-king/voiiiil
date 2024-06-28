import gradio as gr
import cv2
import numpy as np
from PIL import Image
from model import Model

# Load the model
def get_predictor_model():
    model = Model()
    return model

model = get_predictor_model()

def classify_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = np.array(image)
    prediction = model.predict(image=image)
    label_text = prediction['label'].title()
    return label_text

def process_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame_rgb)
        label = prediction['label']
        conf = prediction['confidence']
        frame_bgr = cv2.putText(frame.copy(), f"{label.title()} ({conf:.2f})", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2, 
                                cv2.LINE_AA)

        cv2.imshow('Real-time Classification', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Gradio interface for image classification (not real-time)
image_interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Textbox(label="Predicted Label"),
    title="Image Classification",
    description="Upload an image to classify whether there is a fight, fire, car crash, or everything is okay."
)

# Launch the camera interface for real-time video processing
def launch_camera_interface():
    process_camera()

# Launch the camera interface
if __name__ == "__main__":
    launch_camera_interface()
