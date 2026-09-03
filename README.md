# 🚦 AI Traffic Intelligence

An end-to-end **AI-powered traffic monitoring and forecasting system** that combines **Computer Vision, YOLO object detection, BoT-SORT tracking, traffic analytics, Machine Learning, automated alerts, and Streamlit** to analyze traffic conditions and predict near-future congestion.

## 📌 Project Overview

Traffic congestion is a major challenge in urban environments. Traditional traffic monitoring systems primarily focus on understanding the current traffic situation.

**AI Traffic Intelligence** extends this approach by combining computer vision and machine learning to:

* Detect and track vehicles from traffic video
* Measure traffic characteristics
* Analyze congestion levels
* Build time-series traffic features
* Forecast near-future congestion
* Generate automated traffic alerts
* Visualize traffic insights through an interactive dashboard

The project demonstrates an end-to-end AI/ML workflow from **video processing → data generation → feature engineering → model training → prediction → alert generation → dashboard visualization**.

---

## 🎯 Key Objectives

* Detect vehicles from traffic video
* Track vehicles across video frames
* Count vehicles by category
* Analyze traffic density within a Region of Interest (ROI)
* Estimate approximate vehicle speed
* Detect vehicle movement direction
* Analyze traffic across different zones
* Calculate congestion scores
* Store historical traffic information
* Create time-series ML features
* Forecast congestion approximately 5 seconds ahead
* Generate automated traffic alerts
* Visualize traffic metrics through Streamlit
* Run the complete workflow using a single pipeline command

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
                      BoT-SORT Tracking
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Vehicle Count    Speed          Direction
                           Estimation        Analysis
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
                    Historical Data Logging
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
                Prediction         Evaluation
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
Frame Processing
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

## 🧠 AI & Computer Vision

### YOLO Object Detection

The system uses **YOLO** for real-time vehicle detection from traffic video.

The detection pipeline can identify common traffic classes such as:

* Cars
* Motorcycles
* Bicycles
* Buses
* Trucks

### BoT-SORT Multi-Object Tracking

**BoT-SORT** is used to maintain vehicle identities across consecutive frames.

Tracking enables the system to associate observations with individual vehicles and supports downstream analysis such as counting, speed estimation, and direction detection.

### Traffic Analytics

The computer vision pipeline generates:

* Vehicle count
* Average vehicle speed
* Traffic density
* Congestion score
* Congestion level
* Vehicle direction
* Zone-wise vehicle distribution

---

## 🤖 Traffic Congestion Engine

The congestion engine combines multiple traffic signals instead of relying only on vehicle count.

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

The prototype categorizes traffic into:

```text
NORMAL
MODERATE
HEAVY
```

This provides a more meaningful representation of traffic conditions than vehicle count alone.

---

## 🔮 Future Traffic Forecasting

Traffic observations are aggregated into **1-second intervals** and transformed into time-series features.

### Features

* Current vehicle count
* Current average speed
* Previous vehicle counts
* Previous speed values
* Congestion score history
* Rolling averages
* Vehicle count changes
* Speed changes

### Forecasting Target

The model predicts the traffic congestion level approximately **5 seconds into the future**.

The forecasting workflow uses a **time-based train/test split**, preventing random mixing of past and future observations.

---

## 🧪 Machine Learning Models

Three classification algorithms were evaluated:

### Logistic Regression

Used as the baseline classification model.

### Random Forest

Used to capture nonlinear relationships between traffic features.

### Gradient Boosting

Used as an additional ensemble learning approach.

The current pipeline selected **Random Forest** as the stored forecasting model.

### Evaluation

The current prototype uses a very small demonstration dataset, so evaluation metrics should be interpreted as **pipeline validation rather than production-level model performance**.

---

## 📊 Prototype Results

The current demonstration uses a **39-second traffic video**.

```text
Video Duration       : 39 seconds
Frame Rate            : 30 FPS
Total Frames          : 1170
1-Second Observations : 39
```

### Frame-Level Congestion Distribution

```text
NORMAL      897
MODERATE    263
HEAVY        10
```

### Forecasting Evaluation

```text
Test Samples: 6

Random Forest:
Accuracy          : 0.83
Balanced Accuracy : 0.50
```

Because the test set is extremely small and the classes are imbalanced, the reported accuracy **should not be interpreted as real-world forecasting accuracy**.

The results demonstrate that the complete data-processing, feature-engineering, training, evaluation, and prediction pipeline is functioning correctly.

---

## 🚨 Traffic Alert System

The alert engine converts predicted traffic conditions into actionable information.

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

Alert levels and recommendations are generated according to the predicted traffic condition.

---

## 📊 Streamlit Dashboard

The interactive Streamlit dashboard provides:

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

| Category         | Technologies                                          |
| ---------------- | ----------------------------------------------------- |
| Programming      | Python                                                |
| Computer Vision  | OpenCV, YOLO, BoT-SORT                                |
| Data Processing  | NumPy, Pandas                                         |
| Machine Learning | Scikit-learn                                          |
| ML Models        | Logistic Regression, Random Forest, Gradient Boosting |
| Visualization    | Matplotlib, Streamlit                                 |
| Model Storage    | Joblib                                                |
| Data Storage     | CSV                                                   |
| Version Control  | Git, GitHub                                           |

---

## 📁 Project Structure

```text
AI-Traffic-Intelligence/
│
├── data/
│   └── images/
│       └── traffic.jpg
│
├── models/
│   └── traffic_forecasting_model.pkl
│
├── src/
│   ├── main.py
│   ├── create_test_video.py
│   ├── read_video.py
│   ├── extract_frames.py
│   ├── inspect_frame.py
│   ├── yolo_detect.py
│   ├── track_vehicles.py
│   ├── count_vehicles.py
│   ├── traffic_density.py
│   ├── inspect_video.py
│   ├── show_roi_frame.py
│   ├── congestion_detection.py
│   ├── speed_estimation.py
│   ├── direction_detection.py
│   ├── zone_analysis.py
│   ├── improved_congestion.py
│   ├── traffic_logger.py
│   ├── analyze_traffic_data.py
│   ├── prepare_ml_data.py
│   ├── train_traffic_model.py
│   ├── train_random_forest.py
│   ├── prepare_future_prediction.py
│   ├── train_future_model.py
│   ├── prepare_traffic_features.py
│   ├── train_advanced_traffic_model.py
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
├── requirements.txt
└── README.md
```

> **Note:** Generated outputs and large traffic video files are excluded from the GitHub repository through `.gitignore`. They can be generated locally by running the project pipeline with appropriate input video data.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/madhu-30505/AI-Traffic-Intelligence.git
cd AI-Traffic-Intelligence
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Complete Pipeline

The complete processing workflow can be executed with:

```powershell
python src\run_pipeline.py
```

The pipeline performs:

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

Open the local Streamlit URL displayed in the terminal.

---

## ⚠️ Limitations

This project is currently a **prototype / portfolio demonstration**.

### Dataset Size

The current source video contains only 39 seconds of traffic footage.

This is insufficient for reliable real-world traffic forecasting.

### Speed Estimation

The prototype uses an assumed pixel-to-distance calibration:

```text
600 pixels ≈ 20 meters
```

Therefore, the calculated speed values are approximate.

A production implementation would require proper camera calibration, road-specific reference measurements, or homography-based distance estimation.

### Forecasting Data

The current forecasting dataset contains only 39 one-second observations and is highly imbalanced.

A production model would require significantly more diverse traffic data collected across different:

* Traffic volumes
* Road conditions
* Weather conditions
* Times of day
* Camera angles
* Traffic patterns

### Vehicle Tracking

Tracking can occasionally experience ID switches or missed detections depending on:

* Video quality
* Camera angle
* Lighting
* Occlusion
* Detection quality

---

## 🚀 Future Improvements

Potential production-level improvements include:

* Collecting several hours or days of traffic video
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
* Automated model retraining
* SMS/email notification integration
* Route recommendation integration

---

## 💡 Key Learning Outcomes

This project provided hands-on implementation of:

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

> **An end-to-end AI traffic monitoring and forecasting prototype combining YOLO-based computer vision, BoT-SORT vehicle tracking, traffic analytics, machine learning forecasting, automated alerts, and an interactive Streamlit dashboard.**
