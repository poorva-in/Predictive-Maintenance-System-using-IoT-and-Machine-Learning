# Predictive-Maintenance-System-using-IoT-and-Machine-Learning
A smart predictive maintenance system developed using Machine Learning and IoT technologies to monitor industrial machine health and detect faults before failure occurs.

# Overview
This project focuses on predicting machine failures using sensor data and machine learning techniques. The system analyzes important machine parameters such as:

- Air Temperature
- Process Temperature
- Rotational Speed (RPM)
- Torque
- Tool Wear

The machine condition is classified into:

- Normal
- Warning
- Fault

The project also integrates cloud monitoring and real-time alert systems using ThingSpeak, SMS, and Email notifications.

# Features
- Machine condition prediction using KNN algorithm
- Real-time IoT cloud monitoring with ThingSpeak
- SMS alerts using Twilio API
- Email notification system
- Root cause analysis for detected faults
- Data balancing using SMOTE
- Simulated ESP32 sensor environment using Python
- Visualization of machine health and sensor data

# Technologies Used
- Python
- Machine Learning
- K-Nearest Neighbors (KNN)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- ThingSpeak
- Twilio API
- SMOTE Technique

# Dataset
The project uses the AI4I 2020 Predictive Maintenance Dataset from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

# Working Process
- Collect machine sensor data
- Preprocess and clean the dataset
- Balance the dataset using SMOTE
- Train the KNN machine learning model
- Predict machine condition
- Send data to ThingSpeak cloud
- Generate SMS and Email alerts for abnormal conditions

# Model Performance
- Algorithm Used: K-Nearest Neighbors (KNN)
- Accuracy Achieved: 94.17%
- Successfully classified machine conditions into Normal, Warning, and Fault categories

# IoT Integration
The system sends real-time machine data to the ThingSpeak cloud platform for monitoring and visualization.
The project simulates ESP32-based sensor communication using Python.

# Alert System
When abnormal machine conditions are detected:

SMS alerts are sent using Twilio
Email notifications are triggered automatically
Maintenance recommendations are provided
