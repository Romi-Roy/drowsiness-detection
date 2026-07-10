# SafeDrive - AI Powered Driver Drowsiness Detection & Parking Assistance

SafeDrive is an end-to-end driver safety system that detects drowsiness in real time and proactively assists drivers in finding a safe place to park before fatigue leads to an accident.

## Problem

According to road safety studies, driver fatigue is one of the leading causes of road accidents. Most existing systems only warn the driver with an alarm but fail to provide any practical next step after drowsiness is detected.

SafeDrive addresses this limitation by combining real-time drowsiness detection with intelligent parking assistance.

---

## Solution

The project consists of two connected modules:

### 1. Drowsiness Detection Module

Built using Python, OpenCV, and Google's MediaPipe FaceMesh.

The module continuously analyzes the driver's face using a camera and detects:

- Eye Aspect Ratio (EAR)
- Eye closure duration
- Head nodding
- Driver alertness level

The driver is classified into three states:

- Awake
- Sleepy
- Extremely Sleepy

Whenever drowsiness is detected:

- An audio alarm is triggered.
- The driver's current state is uploaded to Firebase Realtime Database.
- Alertness information is synchronized in real time.

---

### 2. Smart Parking Assistance Module

The Android application continuously listens to Firebase.

Once the driver is classified as:

- **Sleepy:** The app recommends nearby parking locations where the driver can safely take a break.
- **Extremely Sleepy:** A deep learning parking detection model prioritizes the nearest safe roadside parking spot for immediate stopping.

This transforms a simple warning system into an actionable safety solution.

---

## Workflow

Camera
↓
MediaPipe Face Landmark Detection
↓
Eye Aspect Ratio (EAR) + Head Nodding Analysis
↓
Driver State Classification
(Awake / Sleepy / Extremely Sleepy)
↓
Alarm Trigger
↓
Firebase Realtime Database
↓
Android Application
↓
Parking Recommendation Model
↓
Nearest Safe Parking Suggestion

---

## Tech Stack

### Computer Vision

- Python
- OpenCV
- MediaPipe FaceMesh
- NumPy

### Cloud

- Firebase Realtime Database
- Firebase Admin SDK

### Mobile

- Android

### Machine Learning

- Facial Landmark Detection
- Eye Aspect Ratio (EAR)
- Head Movement Analysis
- Parking Recommendation Model

---

## Features

- Real-time face tracking
- Eye Aspect Ratio (EAR) based drowsiness detection
- Head nodding detection
- Audio alert system
- Firebase cloud synchronization
- Android app integration
- Intelligent parking assistance
- Modular architecture for future vehicle integration

---

## Future Improvements

- Driver identification using Face Recognition
- GPS-based parking recommendations
- Steering behavior analysis
- Vehicle CAN bus integration
- Night vision support
- Edge deployment on Raspberry Pi / NVIDIA Jetson

---

## Project Status

Prototype completed as part of an academic project.

The system successfully demonstrates the complete workflow from driver monitoring to intelligent parking assistance using cloud-based communication.
