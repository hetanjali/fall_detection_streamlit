import cv2
import requests
import streamlit as st
import tempfile

# ============================
# CONFIGURATION
# ============================
API_KEY = "We5taABdsOMILDXzW866"
PROJECT_ID = "fall-detection-mbldh"
MODEL_VERSION = "1"

# Roboflow REST API endpoint for inference
INFERENCE_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={API_KEY}"

# ============================
# FUNCTION TO RUN INFERENCE
# ============================
def infer_frame(frame, conf_threshold):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(
        INFERENCE_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "video_frame"}
    )
    if response.status_code == 200:
        preds = response.json()
        if "predictions" in preds:
            return [p for p in preds["predictions"] if p["confidence"] >= conf_threshold]
    return []


def annotate_frame(frame, predictions):
    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        class_name = pred["class"].lower()
        conf = pred["confidence"]

        # Flip the labels as per your requirement
        if class_name == "fall":
            label, color = "stand", (0, 255, 0)
        elif class_name == "stand":
            label, color = "fall", (0, 0, 255)
        else:
            label, color = class_name, (255, 255, 0)

        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x - w//2, y - h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame


# ============================
# STREAMLIT APP
# ============================
st.title("Fall Detection Demo (Roboflow + Streamlit)")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)

option = st.radio("Choose input source:", ["Upload Video", "Webcam"])

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predictions = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame, predictions)

            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

elif option == "Webcam":
    st.info("Webcam support depends on your environment.")
    # Add webcam logic here if needed