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
## Data Exploration
<img width="2233" height="1328" alt="data_exploration (2)" src="https://github.com/user-attachments/assets/00ee9e12-488a-4278-89bf-2a39638430ba" />

## Confusion Matrix
<img width="989" height="732" alt="confusion_matrix (1)" src="https://github.com/user-attachments/assets/e8efc737-8eab-482e-be14-35d3eabf133c" />

## Decision Boundary
<img width="2085" height="946" alt="decision_boundary (1)" src="https://github.com/user-attachments/assets/cd76ffb8-1b0e-4230-96e5-c2cec52cf13c" />

## ESP 32 Simulation
<img width="2084" height="1769" alt="esp32_simulation (1)" src="https://github.com/user-attachments/assets/93b2fe90-5e9e-4991-b033-fa6079996316" />

# ThingSpeak setup
To enable cloud monitoring, a ThingSpeak channel was created.
# Steps to Get API Key and Channel ID
1. Create an account on ThingSpeak.
2. Create a new channel.
3. Add required fields such as:
   - Air Temperature
   - Process Temperature
   - RPM
   - Torque
   - Tool Wear
   - Alert Flag
4. Open the channel settings.
5. Copy:
   - Write API Key
   - Channel ID
These credentials are then used in the Python code for sending real-time sensor data to the ThingSpeak cloud platform.


Example:

- python
API_KEY = "YOUR_THINGSPEAK_API_KEY"
CHANNEL_ID = "YOUR_CHANNEL_ID"


## ThingSpeak graphs
<img width="544" height="295" alt="graph1" src="https://github.com/user-attachments/assets/3a984d9a-9245-49c4-9151-1c987f23cad7" />
<img width="551" height="281" alt="graph2" src="https://github.com/user-attachments/assets/af633b70-92f7-43c5-85ff-74012d6c8efa" />
<img width="528" height="305" alt="graph3" src="https://github.com/user-attachments/assets/1ae85d60-842a-41af-8c21-0e19ac369b2f" />
<img width="541" height="315" alt="graph4" src="https://github.com/user-attachments/assets/da0efd01-8b58-4611-9162-69202d0b964c" />
<img width="545" height="296" alt="graph5" src="https://github.com/user-attachments/assets/fd9247d2-fa01-44b5-ad27-455f09472ad1" />
<img width="557" height="295" alt="graph6" src="https://github.com/user-attachments/assets/96562ddf-450e-4e5a-8d97-8554ed766b41" />
<img width="570" height="302" alt="graph7" src="https://github.com/user-attachments/assets/7c608dcb-f18e-493b-8cab-c47c0368590e" />

# Alert System
When abnormal machine conditions are detected:

- SMS alerts are sent using Twilio
- Email notifications are triggered automatically
- Maintenance recommendations are provided

# Twilio SMS Alert Configuration
The system uses the Twilio API to send real-time SMS alerts whenever abnormal machine conditions are detected.

# Steps to Configure Twilio
1. Create an account on Twilio.
2. Verify your mobile number.
3. Obtain:
   - Account SID
   - Auth Token
   - Twilio Phone Number
4. Install the Twilio library in Python.

- bash
pip install twilio

# Twilio Phone Number Generation 
The Twilio phone number used in the project was generated through the Twilio cloud platform after creating a Twilio account.

# Steps Followed

1. Created an account on Twilio.
2. Verified the personal mobile number.
3. Accessed the Twilio Console Dashboard.
4. Navigated to:
   - Phone Numbers → Manage → Buy a Number (You dont need to buy a number)
5. Generated a Twilio phone number for SMS services.
6. Used the generated number in the Python code for sending alert messages.

Example:

- python
from_ = "YOUR_TWILIO_PHONE_NUMBER"
to = "YOUR_PHONE_NUMBER"

## SMS alerts
<img width="720" height="1600" alt="sms_alerts" src="https://github.com/user-attachments/assets/e69c0e45-b1b0-4e3e-bb80-e28d8437cb6d" />

# Email Alert Configuration

The system uses Gmail SMTP services to send automated email alerts whenever abnormal machine conditions are detected.

# Steps to Configure Email Alerts

1. Created a Gmail account for sending alerts.
2. Enabled 2-Step Verification in the Google account settings.
3. Generated a Gmail App Password by navigating to:
   - Google Account → Security → App Passwords
4. Selected:
   - App Type → Mail
   - Device → Custom Device
5. Generated a 16-character App Password.
6. Used the generated App Password and email address in the Python code.

Example:

- python
SENDER_EMAIL = "Your email"
SENDER_PASSWORD = "Your gmail app password"
RECEIVER_EMAIL = "Enter receiver email"

## Email alerts 
<img width="1648" height="786" alt="emailalerts" src="https://github.com/user-attachments/assets/c18c92f8-7ac0-4a65-88d2-28deb4608502" />

