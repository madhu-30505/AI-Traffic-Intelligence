# 🚦 AI Traffic Intelligence

An end-to-end **AI-powered traffic monitoring and forecasting system** that uses Computer Vision, YOLO object detection, vehicle tracking, traffic analytics, Machine Learning, automated alerts, and Streamlit to analyze traffic conditions and predict near-future congestion.

---

## 📌 Project Overview

Traffic congestion is a major problem in urban areas. Traditional traffic monitoring systems mainly observe current traffic conditions, while this project combines **Computer Vision and Machine Learning** to analyze traffic and provide an early indication of future congestion.

The system processes traffic video, detects and tracks vehicles, calculates traffic metrics, stores historical traffic data, forecasts future congestion, and generates traffic alerts through an interactive dashboard.

---

## 🎯 Objectives

* Detect vehicles from traffic video.
* Track vehicles across video frames.
* Count vehicles by category.
* Analyze traffic density within a Region of Interest (ROI).
* Estimate approximate vehicle speed.
* Detect vehicle movement direction.
* Analyze traffic across different zones.
* Identify traffic congestion.
* Store historical traffic information.
* Engineer time-series traffic features.
* Forecast congestion approximately 5 seconds ahead.
* Generate automated traffic alerts.
* Visualize traffic information through a Streamlit dashboard.
* Provide a single-command pipeline for the complete workflow.

---

## 🏗️ System Architecture

```text
                    🚦 AI TRAFFIC INTELLIGENCE
                              │
                              ▼
                       Traffic Video
                              │
                              ▼
                           OpenCV
                              │
                              ▼
                     YOLO Object Detection
                              │
                              ▼
                      Vehicle Tracking
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Vehicle Count    Speed          Direction
               │          Estimation        Analysis
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                         ROI Analysis
                              │
                     Traffic Density
                              │
                              ▼
                    Congestion Detection
                              │
                              ▼
                    Historical Data (CSV)
                              │
                              ▼
                    1-Second Aggregation
                              │
                              ▼
                    Feature Engineering
                              │
                              ▼
                  ML Traffic Forecasting
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
                Prediction         Model Evaluation
                     │
                     ▼
                Alert Engine
                     │
                     ▼
             Streamlit Dashboard
```

---

## 🔄 Complete Data Pipeline

```text
Traffic Video
      ↓
Frame Extraction
      ↓
YOLO Vehicle Detection
      ↓
BoT-SORT Vehicle Tracking
      ↓
Vehicle IDs
      ↓
Vehicle Counting
      ↓
ROI Traffic Density
      ↓
Speed Estimation
      ↓
Direction Detection
      ↓
Zone Analysis
      ↓
Congestion Scoring
      ↓
Historical Traffic Logging
      ↓
1-Second Traffic Aggregation
      ↓
Time-Series Feature Engineering
      ↓
5-Second Future Target
      ↓
ML Model Training
      ↓
Model Evaluation
      ↓
Future Congestion Prediction
      ↓
Traffic Alert Generation
      ↓
Streamlit Dashboard
```

---

## 🧠 AI & Machine Learning

### Computer Vision

The system uses **YOLO** for real-time object detection.

Vehicle categories include:

* Cars
* Motorcycles
* Bicycles
* Buses
* Trucks

### Vehicle Tracking

**BoT-SORT** is used to maintain vehicle identities across frames.

This allows the system to associate observations with individual tracked vehicles.

### Traffic Metrics

The system calculates:

* Vehicle count
* Average vehicle speed
* Traffic density
* Congestion score
* Congestion level
* Vehicle direction
* Zone-wise vehicle distribution

---

## 🤖 Traffic Congestion Engine

Traffic congestion is determined using multiple traffic signals rather than vehicle count alone.

The prototype combines:

```text
Vehicle Count
      +
Average Speed
      +
Congestion Duration
      ↓
Congestion Score
      ↓
Traffic Level
```

Traffic levels:

```text
NORMAL
MODERATE
HEAVY
```

---

## 🔮 Future Traffic Forecasting

The system aggregates traffic information into **1-second intervals**.

Time-series features include:

* Current vehicle count
* Current average speed
* Previous vehicle counts
* Previous speed values
* Congestion score history
* Rolling averages
* Vehicle count changes
* Speed changes

The forecasting target is the traffic congestion level approximately **5 seconds into the future**.

---

## 🧪 Machine Learning Models

Three classification models were evaluated:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Used to capture nonlinear relationships between traffic features.

### 3. Gradient Boosting

Used as an additional ensemble learning approach.

The evaluation uses a **time-based train/test split** to avoid randomly mixing past and future observations.

The current pipeline selected **Random Forest** as the stored forecasting model.

---

## 📊 Current Prototype Results

The current demonstration uses a **39-second traffic video**.

Dataset:

```text
Video duration: 39 seconds
Frame rate: 30 FPS
Total frames: 1170
1-second observations: 39
```

Frame-level congestion distribution:

```text
NORMAL      897
MODERATE    263
HEAVY        10
```

The current forecasting dataset is small and highly imbalanced.

Therefore, the current model results should be interpreted as **pipeline validation rather than production-level forecasting performance**.

The latest evaluation produced:

```text
Test samples: 6

Random Forest:
Accuracy:          0.83
Balanced Accuracy: 0.50
```

The 83% accuracy should **not** be interpreted as an 83% real-world prediction accuracy because the test set is extremely small and does not contain a balanced representation of all congestion classes.

---

## 🚨 Traffic Alert System

The alert engine converts model predictions into actionable traffic information.

Example:

```text
Predicted Congestion:
MODERATE

Alert Level:
MEDIUM

Recommendation:
Monitor traffic closely and consider alternate routes
if congestion increases.
```

The system can produce different alert levels depending on predicted traffic conditions.

---

## 📊 Streamlit Dashboard

The dashboard provides:

### Current Traffic Status

* Vehicle count
* Average speed
* Congestion score
* Current congestion level

### AI Future Prediction

* Predicted congestion
* Prediction time
* Model probability
* Class probability visualization

### Traffic Alerts

* Alert level
* Alert message
* Recommendation
* Speed insight
* Volume insight

### Traffic Trends

* Vehicle count trend
* Average speed trend
* Congestion score trend

### Traffic Distribution

* Congestion level distribution
* Congestion score distribution

---

## 🛠️ Technologies Used

### Programming

* Python

### Computer Vision

* OpenCV
* YOLO
* BoT-SORT

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* Gradient Boosting

### Visualization

* Matplotlib
* Streamlit

### Model & Data Storage

* Joblib
* CSV

### Development

* Git
* GitHub
* Python Virtual Environment

---

## 📁 Project Structure

```text
AI-Traffic-Intelligence/
│
├── data/
│   ├── images/
│   │   └── traffic.jpg
│   ├── videos/
│   │   └── traffic.mp4
│   └── test_traffic.mp4
│
├── models/
│   └── traffic_forecasting_model.pkl
│
├── outputs/
│   ├── traffic_history_detailed.csv
│   ├── traffic_1sec_dataset.csv
│   ├── traffic_1sec_ml_features.csv
│   ├── future_traffic_predictions.csv
│   ├── traffic_alerts.csv
│   └── model_metadata.csv
│
├── src/
│   ├── main.py
│   ├── read_video.py
│   ├── extract_frames.py
│   ├── yolo_detect.py
│   ├── track_vehicles.py
│   ├── count_vehicles.py
│   ├── traffic_density.py
│   ├── congestion_detection.py
│   ├── speed_estimation.py
│   ├── direction_detection.py
│   ├── zone_analysis.py
│   ├── improved_congestion.py
│   ├── traffic_logger.py
│   ├── create_1sec_dataset.py
│   ├── prepare_1sec_ml_features.py
│   ├── train_1sec_forecasting_models.py
│   ├── evaluate_traffic_model.py
│   ├── predict_future_traffic.py
│   ├── traffic_alerts.py
│   ├── dashboard.py
│   └── run_pipeline.py
│
├── tests/
├── docs/
├── notebooks/
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Traffic-Intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn joblib streamlit
```

---

## ▶️ Run the Complete Pipeline

The complete processing pipeline can be executed using a single command:

```powershell
python src\run_pipeline.py
```

This automatically performs:

```text
Traffic Data Collection
        ↓
1-Second Aggregation
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Future Prediction
        ↓
Traffic Alerts
```

---

## 📊 Launch the Dashboard

After running the pipeline:

```powershell
streamlit run src\dashboard.py
```

Then open the local Streamlit address shown in the terminal.

---

## ⚠️ Limitations

This project is currently a **prototype / portfolio demonstration**.

### Dataset Size

The current source video contains only 39 seconds of traffic footage.

This is insufficient for reliable real-world traffic forecasting.

### Speed Estimation

The current prototype uses an assumed pixel-to-distance calibration:

```text
600 pixels ≈ 20 meters
```

Therefore, speed values are approximate.

A production system would require camera calibration and road-specific reference measurements or homography.

### Forecasting Data

The current forecasting dataset contains only 39 one-second observations and is highly imbalanced.

More diverse traffic data is required for meaningful model validation.

### Tracking

Vehicle tracking can occasionally experience ID switches or missed detections depending on video quality, camera angle, lighting, and vehicle occlusion.

---

## 🚀 Future Improvements

Potential production-level improvements include:

* Collecting several hours/days of traffic video
* Multi-camera traffic monitoring
* Lane-level traffic analysis
* Polygon-based road zones
* Camera calibration and homography
* More accurate speed estimation
* Traffic signal state detection
* Weather-aware traffic prediction
* Rush-hour pattern detection
* Longer forecasting horizons
* LSTM/GRU/Transformer-based forecasting
* Real-time CCTV streaming
* Database integration
* Cloud deployment
* Docker containerization
* MLOps monitoring
* Model retraining pipeline
* SMS/email notification integration
* Route recommendation integration

---

## 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Computer Vision
* Object Detection
* Multi-Object Tracking
* OpenCV Video Processing
* ROI Analysis
* Feature Engineering
* Time-Series Data Preparation
* Classification
* Ensemble Machine Learning
* Model Evaluation
* Predictive Analytics
* Automated Alert Generation
* Data Visualization
* Streamlit Application Development
* End-to-End ML Pipeline Design

---

## 👩‍💻 Author

**Madhumitha**

Computer Science Engineering Graduate | AI/ML & Data Science Enthusiast

---

## ⭐ Project Highlights

> **An end-to-end AI traffic monitoring and forecasting prototype combining YOLO-based computer vision, vehicle tracking, traffic analytics, machine learning forecasting, automated alerts, and an interactive Streamlit dashboard.**
