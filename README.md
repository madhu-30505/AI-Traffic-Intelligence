# 🚦 AI Traffic Intelligence

An AI-powered traffic monitoring prototype that uses Computer Vision and Machine Learning to detect vehicles, analyze traffic congestion, and predict future traffic conditions.

## ✨ Features

* Vehicle detection using YOLO.
* Vehicle tracking using BoT-SORT.
* Traffic density and congestion analysis.
* Approximate vehicle speed estimation.
* Traffic forecasting using Machine Learning.
* Congestion alerts and Streamlit dashboard.

## 🛠️ Tech Stack

Python | OpenCV | YOLO | BoT-SORT | Scikit-learn | Pandas | NumPy | Streamlit

## ⚙️ Workflow

1. Process traffic video frames.
2. Detect and track vehicles.
3. Analyze traffic density and congestion.
4. Prepare historical traffic data.
5. Train a Machine Learning model to forecast traffic.
6. Display traffic insights through a dashboard.

## 📊 Model Evaluation

The prototype achieved **83% accuracy** on a test set of just 6 samples. The dataset is extremely small and imbalanced, so this result is only an initial pipeline validation—not evidence of reliable real-world forecasting.

## 🚀 Getting Started

```bash
git clone https://github.com/madhu-30505/AI-Traffic-Intelligence.git
cd AI-Traffic-Intelligence
pip install -r requirements.txt
```

Run the dashboard, if supported by your project setup:

```bash
streamlit run src/dashboard.py
```

## ⚠️ Limitations

* Tested on a short traffic video.
* Speed estimates depend on assumed pixel-to-distance calibration.
* More diverse traffic videos are needed for reliable forecasting.

## 👩‍💻 Author

**Madhumitha M**
Computer Science and Engineering | Aspiring AI/ML Engineer

[GitHub](https://github.com/madhu-30505) | [LinkedIn](https://linkedin.com/in/madhu0505)

