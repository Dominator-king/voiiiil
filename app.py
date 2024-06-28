
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

def process_video(input_video):
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    
    out = cv2.VideoWriter('output.avi', fourcc, fps, size)
    success = True

    while success and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame_rgb)
        label = prediction['label']
        conf = prediction['confidence']
        frame_bgr = cv2.putText(frame, f"{label.title()} ({conf:.2f})", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2, 
                                cv2.LINE_AA)

        out.write(frame_bgr)

    cap.release()
    out.release()
    return 'output.avi'

# Gradio interfaces
image_interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Textbox(label="Predicted Label"),
    title="Image Classification",
    description="Upload an image to classify whether there is a fight, fire, car crash, or everything is okay."
)

video_interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Video Processing",
    description="Upload a video to classify its content frame by frame."
)

# Combine interfaces
demo = gr.TabbedInterface([image_interface, video_interface], ["Image Classification", "Video Processing"])

if __name__ == "__main__":
    demo.launch(share=True)