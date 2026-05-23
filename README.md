# 🧠 Human Fall Detection System

## 🔗 Live Demo
[Click here to try the app live](https://fall-detection-cv-stream-lit.streamlit.app/)

## Overview
A computer vision system that automatically detects human fall 
events in video using deep learning. Built entirely as a 
self-initiated project to explore real-world applications of 
AI in safety and healthcare monitoring.

The system analyzes uploaded videos frame by frame, sends each 
frame to a Roboflow-hosted deep learning model, and classifies 
human posture as either "Fall Detected" or "Standing" in real time.

## What I Built
- Integrated Roboflow's object detection API for pose-based 
  fall classification
- Built a fully interactive web application using Streamlit
- Implemented frame extraction and real-time inference pipeline 
  using OpenCV
- Added confidence threshold control so users can tune 
  detection sensitivity
- Deployed the app live on Streamlit Cloud

## Tech Stack
- Python
- OpenCV — frame extraction and image processing
- Roboflow API — deep learning model inference
- Streamlit — web app interface and deployment
- NumPy — array and image data handling

## How It Works
1. Upload a video (MP4, MOV, AVI)
2. Frames are extracted using OpenCV
3. Each frame is sent to Roboflow model via REST API
4. Model returns fall or standing classification with confidence score
5. Result is displayed live in Streamlit UI

## How to Run Locally
pip install -r requirements.txt

streamlit run fall_detection_streamlit/project_streamlit.py

## Developer
Hetanjali Vaghela

B.E. Robotics & Automation Engineering | LDCE Ahmedabad

