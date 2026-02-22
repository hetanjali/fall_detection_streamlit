# Human Fall Detection System

## Overview
This project detects human fall events in video using a Roboflow-hosted deep learning model integrated via API and deployed using Streamlit.

## Tech Stack
- Python
- Streamlit
- OpenCV
- Roboflow API
- NumPy

## How It Works
1. Upload video
2. Extract frames
3. Send frames to Roboflow model via API
4. Detect fall or standing posture
5. Display result in Streamlit UI

## How to Run

pip install -r requirements.txt

streamlit run project_streamlit.py
