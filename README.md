# Real-time Cloud observation and monitoring equipment(COME)

COME is an end-to-end system for real-time cloud observation, analysis, and monitoring.
The project combines deep learning and IoT hardware to quantify and classify cloud formations for environmental and atmospheric applications.
## Project Overview
This repository contains all core components used in the development of the COME system, including:

Cloud segmentation and quantification

Cloud type classification

Embedded IoT data acquisition and deployment

The system is designed to operate in real time, making it suitable for smart environmental monitoring and edge-AI deployments.
## Core Components
1. UNet-Based Cloud Quantification


Pixel-level cloud segmentation using a UNet architecture


Enables accurate cloud coverage estimation and spatial analysis


2. EfficientNetB0 Cloud Classification


Lightweight EfficientNetB0 variant optimized for edge deployment


Classifies cloud types with high accuracy and low computational cost


3. IoT & Embedded System Development


Hardware integration for image capture and data transmission


Designed for low-power, real-time cloud observation in the field


## Technologies Used


Deep Learning: UNet, EfficientNetB0


Frameworks: Pytorch and TensorFlow / Keras


Computer Vision: OpenCV


Embedded Systems & IoT: Microcontrollers, sensors, edge devices


Programming Languages: Python, C/C++


📌 Use Cases


Smart weather and atmospheric monitoring


Climate and environmental research


Edge-AI vision systems


Academic and industrial IoT deployments


📂 Repository Structure
├── unet_model/          # Cloud segmentation & quantification
├── efficientnet_model/ # Cloud classification
├── iot/                # Embedded & hardware-related code
└── docs/               # Project documentation

# Motivation
Traditional cloud monitoring systems are done manually here in Ghana and heavily based on estimates of professionals. This approach is time-consuming and error-prone.
COME aims to provide a low-cost, scalable, and intelligent alternative using modern deep learning and embedded systems.
r, embedded, or IoT roles specifically

# The developed U-Net model achieved 86.5% segmentation accuracy, while the EfficientNetB0-based classifier achieved 92% classification accuracy when tested on 50 labeled samples using the correct prediction metrics.


## 🎥 Project Demo

[![COME Project Demo](demo/demo.png)](https://www.youtube.com/watch?v=ag4uIiE48iI)
