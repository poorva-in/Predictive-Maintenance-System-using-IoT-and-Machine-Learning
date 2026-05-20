import pandas as pd
df = pd.read_csv('predictive_maintenance.csv')
print(df.columns.tolist())
print(df.head(3))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

print("All the libraries are imported successfully!")

df.rename(columns={
    'Air temperature [K]':     'air_temp',
    'Process temperature [K]': 'process_temp',
    'Rotational speed [rpm]':  'rpm',
    'Torque [Nm]':             'torque',
    'Tool wear [min]':         'tool_wear',
    'Target':                  'failure',
    'Failure Type':            'failure_type',
}, inplace=True)

def assign_label(row):
    if row['failure'] == 1:
      if row['failure_type'] in ['Heat Dissipation Failure', 'Overstrain Failure', 'Power Failure']:
        return 'fault'
      else:
        return 'warning'
    else:
      if row['torque'] > 65 or row['tool_wear'] > 210:
            return 'fault'
      elif row['torque'] > 50 or row['tool_wear'] > 160:
            return 'warning'
      else:
            return 'normal'


df['label'] = df.apply(assign_label, axis=1)

print("Data cleaned successfully!")
print()
print("Class distribution:")
print(df['label'].value_counts())
print()
print("Sample:")
print(df[['air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear', 'label']].head())
colors = {'normal': '#1D9E75', 'warning': '#E8A020', 'fault': '#E24B4A'}


fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Predictive Maintenance — Sensor Data Exploration',
             fontsize=14, fontweight='bold')


# Plot 1: Air Temperature
ax = axes[0, 0]
#pehla row pehla column
for label in ['normal', 'warning', 'fault']:
    ax.hist(df[df['label'] == label]['air_temp'],
            bins=30, alpha=0.7, color=colors[label], label=label)

ax.set_title('Air Temperature (K)')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)

# Plot 2: RPM
ax = axes[0, 1]
for label in ['normal', 'warning', 'fault']:
    ax.hist(df[df['label'] == label]['rpm'],
            bins=30, alpha=0.7, color=colors[label], label=label)
ax.set_title('Rotational Speed (RPM)')
ax.set_xlabel('RPM'); ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)


 # Plot 3: Torque
ax = axes[0, 2]
for label in ['normal', 'warning', 'fault']:
    ax.hist(df[df['label'] == label]['torque'],
            bins=30, alpha=0.7, color=colors[label], label=label)
ax.set_title('Torque (Nm)')
ax.set_xlabel('Torque (Nm)'); ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)


# Plot 4: RPM vs Torque
ax = axes[1, 0]
for label in ['normal', 'warning', 'fault']:
    sub = df[df['label'] == label]
    ax.scatter(sub['rpm'], sub['torque'],
               c=colors[label], label=label,
               alpha=0.4, edgecolors='none', s=15)
ax.set_title('RPM vs Torque')
ax.set_xlabel('RPM'); ax.set_ylabel('Torque (Nm)')
ax.legend(); ax.grid(alpha=0.3)


# Plot 5: Air Temp vs Process Temp
ax = axes[1, 1]
for label in ['normal', 'warning', 'fault']:
    sub = df[df['label'] == label]
    ax.scatter(sub['air_temp'], sub['process_temp'],
               c=colors[label], label=label,
               alpha=0.4, edgecolors='none', s=15)
ax.set_title('Air Temp vs Process Temp')
ax.set_xlabel('Air Temp (K)'); ax.set_ylabel('Process Temp (K)')
ax.legend(); ax.grid(alpha=0.3)


# Plot 6: Tool Wear
ax = axes[1, 2]
for label in ['normal', 'warning', 'fault']:
    ax.hist(df[df['label'] == label]['tool_wear'],
            bins=30, alpha=0.7, color=colors[label], label=label)
ax.set_title('Tool Wear (min)')
ax.set_xlabel('Tool Wear (min)'); ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)


plt.tight_layout()
plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: data_exploration.png")
!pip install imbalanced-learn
FEATURES = ['air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear']
LABEL = 'label'

X = df[FEATURES].values
y = df[LABEL].values
print("X and y defined!")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Before balancing:")
print(pd.Series(y).value_counts())
print()
print("After balancing:")
print(pd.Series(y_balanced).value_counts())
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 # Train KNN model
model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train_scaled, y_train)

print("Model trained: K-Nearest Neighbors (k=5)")
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")
print(f"Training accuracy: {model.score(X_train_scaled, y_train)*100:.2f}%")
print(f"Testing  accuracy: {model.score(X_test_scaled,  y_test)*100:.2f}%")

y_pred = model.predict(X_test_scaled)#2000 samples ko test karega

print("Classification Report:")
print("=" * 55)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['normal', 'warning', 'fault'])#calculating confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['normal', 'warning', 'fault'],
            yticklabels=['normal', 'warning', 'fault'],
            linewidths=0.5, ax=ax)

ax.set_title('Confusion Matrix — Predictive Maintenance',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()#fixes the plot ek sath rakhta hai so that koi kuch overlap na ho
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

NEW_READING = {
    'air_temp':     301,
    'process_temp': 311,
    'rpm':          1400,
    'torque':       50,
    'tool_wear':    160,
}

def predict_machine_status(reading):
    # SAFETY THRESHOLDS (based on dataset statistics)
    x = np.array([[reading['air_temp'], reading['process_temp'],
                   reading['rpm'],      reading['torque'],
                   reading['tool_wear']]])
    x_scaled      = scaler.transform(x)
    prediction    = model.predict(x_scaled)[0]
    probabilities = model.predict_proba(x_scaled)[0]
    confidence    = dict(zip(model.classes_, (probabilities * 100).round(1)))

    icons  = {'normal': '[OK]', 'warning': '[WARN]', 'fault': '[FAULT]'}
    action = {
        'normal':  'Machine is healthy. Continue monitoring.',
        'warning': 'Elevated readings. Schedule maintenance soon.',
        'fault':   'CRITICAL — Stop machine immediately!'
    }

    # Display results
    print("=" * 55)
    print("   PREDICTIVE MAINTENANCE DIAGNOSIS (ROBUST)")
    print("=" * 55)
    print(f"   Air Temp : {reading['air_temp']} K")
    print(f"   Process Temp : {reading['process_temp']} K")
    print(f"   RPM : {reading['rpm']}")
    print(f"   Torque : {reading['torque']} Nm")
    print(f"   Tool Wear : {reading['tool_wear']} min")
    print()

    print(f"   RESULT       : {icons[prediction]}  {prediction.upper()}")
    print(f"   Confidence   : {confidence[prediction]}%")
    print()
    print("   Class Probabilities:")
    for cls in ['normal', 'warning', 'fault']:
        bar = '#' * int(confidence.get(cls, 0) / 5)
        print(f"     {cls:<8}  {confidence.get(cls, 0):>5.1f}%  {bar}")
    print()
    print(f"   Action : {action[prediction]}")
    print()
    print("=" * 55)
    print("   ROOT CAUSE ANALYSIS")
    print("=" * 55)
    causes    = []
    solutions = []

    if reading['torque'] > 60:
        causes.append(f"   TORQUE = {reading['torque']} Nm  (limit: 60 Nm) EXCEEDED!")
        solutions.append("   → Reduce machine load immediately")
        solutions.append("   → Inspect bearing for damage")
    elif reading['torque'] > 45:
        causes.append(f"   TORQUE = {reading['torque']} Nm  (limit: 60 Nm) Elevated")
        solutions.append("   → Monitor torque closely")
        solutions.append("   → Check for mechanical wear")

    if reading['tool_wear'] > 200:
        causes.append(f"   TOOL WEAR = {reading['tool_wear']} min  (limit: 200 min) EXCEEDED!")
        solutions.append("   → Replace cutting tool immediately")
    elif reading['tool_wear'] > 150:
        causes.append(f"   TOOL WEAR = {reading['tool_wear']} min  (limit: 200 min) Elevated")
        solutions.append("   → Plan tool replacement soon")

    if not causes:
        print("   All parameters within safe limits!")
    else:
        print("   Possible Causes:")
        for c in causes:
            print(c)
            print()
            print("   Recommended Solutions:")
        for s in solutions:
              print(s)

    print("=" * 55)
    alert_flag = {'normal': 0, 'warning': 1, 'fault': 2}[prediction]
    print(f"\n   ThingSpeak Alert Flag: {alert_flag}  (0=ok, 1=warning, 2=fault)")

predict_machine_status(NEW_READING)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Decision Boundary — How the AI Classifies Readings',
             fontsize=13, fontweight='bold')

plot_configs = [
    {'ax': axes[0], 'f1': 'rpm',      'f2': 'torque',
     'xl': 'RPM',           'yl': 'Torque (Nm)'},
    {'ax': axes[1], 'f1': 'air_temp', 'f2': 'tool_wear',
     'xl': 'Air Temp (K)',  'yl': 'Tool Wear (min)'},
]

cmap_bg   = ListedColormap(['#A8DFCA', '#F5D78E', '#F5A8A8'])
label_map = {'fault': 0, 'normal': 1, 'warning': 2}

for cfg in plot_configs:
    ax     = cfg['ax']
    f1_idx = FEATURES.index(cfg['f1'])
    f2_idx = FEATURES.index(cfg['f2'])
    X_2d   = X[:, [f1_idx, f2_idx]]

    sc2 = StandardScaler()
    sc2.fit(X_2d)

    x_min = X_2d[:, 0].min() - X_2d[:, 0].std()
    x_max = X_2d[:, 0].max() + X_2d[:, 0].std()
    y_min = X_2d[:, 1].min() - X_2d[:, 1].std()
    y_max = X_2d[:, 1].max() + X_2d[:, 1].std()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    knn2 = KNeighborsClassifier(n_neighbors=5)
    knn2.fit(sc2.transform(X_2d), y)
    Z = np.array([label_map[l] for l in
                  knn2.predict(sc2.transform(np.c_[xx.ravel(), yy.ravel()]))
                  ]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)

    sample = df.sample(min(500, len(df)), random_state=42)
    for label, color in colors.items():
        sub = sample[sample['label'] == label]
        ax.scatter(sub[cfg['f1']], sub[cfg['f2']],
                   c=color, label=label, alpha=0.6,
                   edgecolors='white', linewidths=0.5, s=20, zorder=3)
ax.set_xlabel(cfg['xl'], fontsize=11)
ax.set_ylabel(cfg['yl'], fontsize=11)
ax.set_title(f"{cfg['xl']} vs {cfg['yl']}", fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.2)

patches = [
    mpatches.Patch(color='#A8DFCA', label='Normal zone'),
    mpatches.Patch(color='#F5D78E', label='Warning zone'),
    mpatches.Patch(color='#F5A8A8', label='Fault zone'),
]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: decision_boundary.png")

np.random.seed(42)
N = 200

time_steps, vib_signal  = [], []
temp_signal, rpm_signal = [], []
rms_signal,  pred_labels = [], []

for i in range(N):
    if i < 70:
        vib_amp, temp_base, rpm_base = 0.5,  300.5, 1500
    elif i < 130:
        vib_amp, temp_base, rpm_base = 1.9,  305.0, 1300
    else:
        vib_amp, temp_base, rpm_base = 4.5,  310.0, 1100

    t   = i * 0.01
    vib = abs(vib_amp * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.1))
    tmp = temp_base + np.random.normal(0, 0.5)
    rpm = int(rpm_base + np.random.normal(0, 20))
    rms = float(np.sqrt(vib ** 2))
    torque_est    = 40 + (vib * 5)
    tool_wear_est = i * 1.1
    x_sc = scaler.transform([[tmp, tmp + 10, rpm, torque_est, tool_wear_est]])
    pred = model.predict(x_sc)[0]

    time_steps.append(i)
    vib_signal.append(vib)
    temp_signal.append(tmp)
    rpm_signal.append(rpm)
    rms_signal.append(rms)
    pred_labels.append(pred)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle('ESP32 Simulated Signal — Machine Degradation Over Time',
             fontsize=13, fontweight='bold')
for ax in axes:
    ax.axvspan(0,   70,  alpha=0.07, color='#1D9E75')
    ax.axvspan(70,  130, alpha=0.07, color='#E8A020')
    ax.axvspan(130, 200, alpha=0.07, color='#E24B4A')
    ax.axvline(70,  color='#E8A020', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(130, color='#E24B4A', linestyle='--', linewidth=1, alpha=0.6)

axes[0].plot(time_steps, vib_signal,  color='#378ADD', linewidth=1.5)
axes[0].set_ylabel('Vibration (g)');  axes[0].set_title('Vibration Signal');        axes[0].grid(alpha=0.3)
axes[1].plot(time_steps, temp_signal, color='#E24B4A', linewidth=1.5)
axes[1].set_ylabel('Air Temp (K)');   axes[1].set_title('Temperature');              axes[1].grid(alpha=0.3)
axes[2].plot(time_steps, rpm_signal,  color='#9B59B6', linewidth=1.5)
axes[2].set_ylabel('RPM');            axes[2].set_title('Rotational Speed');          axes[2].grid(alpha=0.3)
axes[3].plot(time_steps, rms_signal,  color='#E8A020', linewidth=1.5)
axes[3].set_ylabel('RMS (g)');        axes[3].set_title('RMS Vibration — Anomaly Indicator'); axes[3].grid(alpha=0.3)
axes[3].set_xlabel('Time Step (simulated ESP32 readings)')

axes[3].text(35,  max(rms_signal)*0.85, 'NORMAL',  color='#1D9E75', fontweight='bold', ha='center', fontsize=11)
axes[3].text(100, max(rms_signal)*0.85, 'WARNING', color='#E8A020', fontweight='bold', ha='center', fontsize=11)
axes[3].text(165, max(rms_signal)*0.85, 'FAULT',   color='#E24B4A', fontweight='bold', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('esp32_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: esp32_simulation.png")

x_sc = scaler.transform([[
    NEW_READING['air_temp'],    NEW_READING['process_temp'],
    NEW_READING['rpm'],         NEW_READING['torque'],
    NEW_READING['tool_wear']
]])
prediction   = model.predict(x_sc)[0]
probs        = model.predict_proba(x_sc)[0]
confidence   = round(float(max(probs)) * 100, 1)
alert_flag   = {'normal': 0, 'warning': 1, 'fault': 2}[prediction]

API_KEY    = "UYYWK3DCZ080ILG8"
CHANNEL_ID = "3307443"
print("=" * 60)
print("  ThingSpeak Cloud Payload")
print("  (This is what ESP32 sends over WiFi)")
print("=" * 60)
print(f"  api_key    = {API_KEY}")
print(f"  field1     = {NEW_READING['air_temp']}    (air temperature K)")
print(f"  field2     = {NEW_READING['process_temp']}   (process temperature K)")
print(f"  field3     = {NEW_READING['rpm']}     (RPM)")
print(f"  field4     = {NEW_READING['torque']}     (torque Nm)")
print(f"  field5     = {NEW_READING['tool_wear']}     (tool wear min)")
print(f"  field6     = {alert_flag}         (0=ok  1=warning  2=fault)")
print(f"  field7     = {confidence}     (model confidence %)")
print(f"  status     = {prediction.upper()}")
print("=" * 60)

import urllib.request

# Send data to ThingSpeak
url = (f"https://api.thingspeak.com/update"
       f"?api_key=UYYWK3DCZ080ILG8"
       f"&field1={NEW_READING['air_temp']}"
       f"&field2={NEW_READING['process_temp']}"
       f"&field3={NEW_READING['rpm']}"
       f"&field4={NEW_READING['torque']}"
       f"&field5={NEW_READING['tool_wear']}"
       f"&field6={alert_flag}"
       f"&field7={confidence}")

response = urllib.request.urlopen(url)
entry_id = response.read().decode()

if entry_id != '0':
    print(f"Data sent to ThingSpeak successfully!")
    print(f"Entry ID: {entry_id}")
    print(f"View your channel: https://thingspeak.com/channels/3307443")
else:
    print("Failed to send. Check your API key.")

import urllib.request
import time


test_readings = [
    # Normal readings
    {'air_temp':300,'process_temp':310,'rpm':1500,'torque':30,'tool_wear':50},
    {'air_temp':300,'process_temp':311,'rpm':1480,'torque':35,'tool_wear':80},
    {'air_temp':301,'process_temp':311,'rpm':1490,'torque':38,'tool_wear':100},
    {'air_temp':301,'process_temp':312,'rpm':1470,'torque':40,'tool_wear':120},
    # Warning readings
    {'air_temp':302,'process_temp':312,'rpm':1300,'torque':46,'tool_wear':155},
    {'air_temp':303,'process_temp':313,'rpm':1250,'torque':50,'tool_wear':165},
    {'air_temp':303,'process_temp':314,'rpm':1200,'torque':53,'tool_wear':175},
    {'air_temp':304,'process_temp':314,'rpm':1150,'torque':57,'tool_wear':185},
    # Fault readings
    {'air_temp':305,'process_temp':315,'rpm':1100,'torque':62,'tool_wear':205},
    {'air_temp':306,'process_temp':316,'rpm':1050,'torque':67,'tool_wear':215},
    {'air_temp':307,'process_temp':316,'rpm':1000,'torque':72,'tool_wear':225},
    {'air_temp':308,'process_temp':317,'rpm':950, 'torque':78,'tool_wear':240},
]

API_KEY = "UYYWK3DCZ080ILG8"

for i, reading in enumerate(test_readings):
    # Alert flag decide karo
    if reading['torque'] > 60 or reading['tool_wear'] > 200:
        alert = 2
    elif reading['torque'] > 45 or reading['tool_wear'] > 150:
        alert = 1
    else:
        alert = 0

    url = (f"https://api.thingspeak.com/update"
           f"?api_key={"UYYWK3DCZ080ILG8"}"
           f"&field1={reading['air_temp']}"
           f"&field2={reading['process_temp']}"
           f"&field3={reading['rpm']}"
           f"&field4={reading['torque']}"
           f"&field5={reading['tool_wear']}"
           f"&field6={alert}")

    response = urllib.request.urlopen(url)
    entry_id = response.read().decode()
    print(f"Reading {i+1}/12 sent — Alert: {alert} — Entry ID: {entry_id}")
    time.sleep(16)  # ThingSpeak free = 15 sec gap

print("Done! Check ThingSpeak now!")

!pip install twilio
from twilio.rest import Client

#account_sid = 'put your account sid'
auth_token = 'enter your authentication token'

client = Client(account_sid, auth_token)

def send_sms(message):
    client.messages.create(
        body=message,
        from_='+12605298698',
        to='+919954957194'
    )

x = [[
    NEW_READING['air_temp'],
    NEW_READING['process_temp'],
    NEW_READING['rpm'],
    NEW_READING['torque'],
    NEW_READING['tool_wear']
]]
x_scaled = scaler.transform(x)

# Your model prediction
prediction = model.predict(x_scaled)[0]
print("Prediction:", prediction)

if prediction == 'fault':
    msg = "FAULT 🚨: Machine MCH-01\nImmediate action required!"
    send_sms(msg)

elif prediction == 'warning':
    msg = "WARNING ⚠️: Machine parameters unstable"
    send_sms(msg)

else:
    print("No SMS sent (Normal condition)")
    #msg = "NORMAL ✅: Machine MCH-01 "

send_sms(msg)


last_status = None

prediction = model.predict(x_scaled)[0]

if prediction == 'fault':
    status = "FAULT"
elif prediction == 'warning':
    status = "WARNING"
else:
    status = "NORMAL"

if status != last_status:
    #send_sms(f"Machine MCH-01 Status: {status}")
    last_status = status

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ─── CONFIG ───────────────────────────────
SENDER_EMAIL    = "poorvarao361@gmail.com"      # your Gmail
SENDER_PASSWORD = "qqoo dbom vmbj lmyy"    # Gmail App Password
RECEIVER_EMAIL  = "raopoorva6@gmail.com"  # who gets the alert
MACHINE_ID      = "MCH-01"
# ──────────────────────────────────────────

def send_email_alert(prediction, reading, confidence_pct):
    if prediction == 'normal':
        print("Normal condition — no email sent.")
        return

    subject_map = {
        'warning': f"⚠️ WARNING — Machine {MACHINE_ID} needs attention",
        'fault':   f"🚨 CRITICAL FAULT — Machine {MACHINE_ID} STOP IMMEDIATELY"
    }
    color_map = {
        'warning': '#E8A020',
        'fault':   '#E24B4A'
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject_map[prediction]
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECEIVER_EMAIL

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
        <div style="max-width: 500px; margin: auto; background: white;
                    border-radius: 10px; padding: 25px;
                    border-top: 5px solid {color_map[prediction]};">

          <h2 style="color: {color_map[prediction]}; margin-top: 0;">
            {'⚠️ WARNING ALERT' if prediction == 'warning' else '🚨 CRITICAL FAULT ALERT'}
          </h2>

          <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="background:#f9f9f9;">
              <td style="padding:8px; border:1px solid #ddd;"><b>Machine ID</b></td>
              <td style="padding:8px; border:1px solid #ddd;">{MACHINE_ID}</td>
            </tr>
            <tr>
              <td style="padding:8px; border:1px solid #ddd;"><b>Status</b></td>
              <td style="padding:8px; border:1px solid #ddd; color:{color_map[prediction]};"><b>{prediction.upper()}</b></td>
            </tr>
            <tr style="background:#f9f9f9;">
              <td style="padding:8px; border:1px solid #ddd;"><b>Confidence</b></td>
              <td style="padding:8px; border:1px solid #ddd;">{confidence_pct}%</td>
            </tr>
            <tr>
              <td style="padding:8px; border:1px solid #ddd;"><b>Timestamp</b></td>
              <td style="padding:8px; border:1px solid #ddd;">{timestamp}</td>
            </tr>
          </table>

          <h3 style="color:#333;">Sensor Readings:</h3>
          <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="background:#f9f9f9;">
              <td style="padding:8px; border:1px solid #ddd;">Air Temperature</td>
              <td style="padding:8px; border:1px solid #ddd;">{reading['air_temp']} K</td>
            </tr>
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">Process Temperature</td>
              <td style="padding:8px; border:1px solid #ddd;">{reading['process_temp']} K</td>
            </tr>
            <tr style="background:#f9f9f9;">
              <td style="padding:8px; border:1px solid #ddd;">RPM</td>
              <td style="padding:8px; border:1px solid #ddd;">{reading['rpm']}</td>
            </tr>
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">Torque</td>
              <td style="padding:8px; border:1px solid #ddd;">{reading['torque']} Nm</td>
            </tr>
            <tr style="background:#f9f9f9;">
              <td style="padding:8px; border:1px solid #ddd;">Tool Wear</td>
              <td style="padding:8px; border:1px solid #ddd;">{reading['tool_wear']} min</td>
            </tr>
          </table>

          <div style="background:{color_map[prediction]}22; border-left: 4px solid {color_map[prediction]};
                      padding: 12px; border-radius: 4px;">
            <b>Recommended Action:</b><br>
            {'Schedule maintenance soon. Monitor parameters closely.' if prediction == 'warning'
             else 'STOP the machine immediately and inspect for damage!'}
          </div>

          <p style="color:#999; font-size:12px; margin-top:20px;">
            This is an automated alert from your Predictive Maintenance System.
          </p>
        </div>
      </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print(f"✅ Email alert sent! Status: {prediction.upper()} at {timestamp}")
    except Exception as e:
        print(f"❌ Email failed: {e}")


# ─── Run it with your NEW_READING ─────────
x = [[
    NEW_READING['air_temp'],    NEW_READING['process_temp'],
    NEW_READING['rpm'],         NEW_READING['torque'],
    NEW_READING['tool_wear']
]]
x_scaled     = scaler.transform(x)
prediction   = model.predict(x_scaled)[0]
probs        = model.predict_proba(x_scaled)[0]
confidence   = round(float(max(probs)) * 100, 1)

send_email_alert(prediction, NEW_READING, confidence)

df.to_csv('cleaned_dataset.csv', index=False)

pd.DataFrame({
    'time_step':   time_steps,
    'vibration':   vib_signal,
    'temperature': temp_signal,
    'rpm':         rpm_signal,
    'rms':         rms_signal,
    'prediction':  pred_labels
}).to_csv('simulation_log.csv', index=False)


print("=" * 55)
print("  cleaned_dataset.csv    — processed dataset")
print("  simulation_log.csv     — ESP32 simulation log")
print("  data_exploration.png   — sensor data graphs")
print("  confusion_matrix.png   — model accuracy")
print("  decision_boundary.png  — AI classification zones")
print("  esp32_simulation.png   — signal simulation")
print()
print(f"  Final Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  Algorithm      : K-Nearest Neighbors (k=5)")
print(f"  Dataset        : AI4I 2020 Predictive Maintenance (Kaggle)")
print(f"  Training rows  : {len(X_train)}")
print("=" * 55)
