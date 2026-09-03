import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

TRAFFIC_FILE = OUTPUTS_DIR / "traffic_1sec_dataset.csv"
PREDICTION_FILE = OUTPUTS_DIR / "future_traffic_predictions.csv"
ALERT_FILE = OUTPUTS_DIR / "traffic_alerts.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(file_path):

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)

    except Exception as error:
        st.error(
            f"Error reading {file_path.name}: {error}"
        )

        return None


traffic_data = load_data(TRAFFIC_FILE)
prediction_data = load_data(PREDICTION_FILE)
alert_data = load_data(ALERT_FILE)


# ============================================================
# HEADER
# ============================================================

st.title("🚦 AI Traffic Intelligence")

st.write(
    "Computer Vision + Vehicle Tracking + Machine Learning "
    "Traffic Monitoring System"
)


# ============================================================
# CHECK TRAFFIC DATA
# ============================================================

if traffic_data is None or traffic_data.empty:

    st.error(
        "Traffic dataset is not available."
    )

    st.stop()


# ============================================================
# LATEST TRAFFIC DATA
# ============================================================

latest = traffic_data.iloc[-1]

current_time = latest["video_time_seconds"]

vehicle_count = float(
    latest["vehicle_count"]
)

average_speed = float(
    latest["average_speed_kmh"]
)

congestion_score = float(
    latest["congestion_score"]
)

congestion_level = str(
    latest["congestion_level"]
)


# ============================================================
# CURRENT TRAFFIC STATUS
# ============================================================

st.markdown("## 📊 Current Traffic Status")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🚗 Vehicles",
        f"{vehicle_count:.2f}"
    )


with col2:

    st.metric(
        "🏎️ Average Speed",
        f"{average_speed:.2f} km/h"
    )


with col3:

    st.metric(
        "🎯 Congestion Score",
        f"{congestion_score:.2f}"
    )


with col4:

    st.metric(
        "🚦 Congestion",
        congestion_level
    )


# ============================================================
# CONGESTION STATUS
# ============================================================

if congestion_level.upper() == "HEAVY":

    st.error(
        "🔴 HEAVY TRAFFIC DETECTED"
    )

elif congestion_level.upper() == "MODERATE":

    st.warning(
        "🟠 MODERATE TRAFFIC DETECTED"
    )

else:

    st.success(
        "🟢 TRAFFIC CONDITIONS ARE NORMAL"
    )


# ============================================================
# AI FUTURE PREDICTION
# ============================================================

st.markdown("## 🔮 AI Future Traffic Prediction")


if prediction_data is not None and not prediction_data.empty:

    prediction = prediction_data.iloc[-1]

    predicted_congestion = str(
        prediction["predicted_congestion"]
    )

    prediction_time = prediction[
        "prediction_time_seconds"
    ]

    model_probability = prediction[
        f"{predicted_congestion.lower()}_probability"
    ] if (
        f"{predicted_congestion.lower()}_probability"
        in prediction_data.columns
    ) else prediction.get(
        "moderate_probability",
        0
    )

    st.info(
        f"AI predicts **{predicted_congestion} congestion** "
        f"at approximately **{prediction_time} seconds**."
    )

    st.metric(
        "🤖 Model Confidence",
        f"{float(model_probability) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    st.markdown(
        "### Prediction Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Congestion Level": [
                "NORMAL",
                "MODERATE",
                "HEAVY"
            ],
            "Probability": [
                prediction["normal_probability"],
                prediction["moderate_probability"],
                prediction["heavy_probability"]
            ]
        }
    )

    probability_df = probability_df.set_index(
        "Congestion Level"
    )

    st.bar_chart(
        probability_df
    )

else:

    st.warning(
        "Future prediction data is not available."
    )


# ============================================================
# TRAFFIC ALERT
# ============================================================

st.markdown("## 🚨 Traffic Alert")


if alert_data is not None and not alert_data.empty:

    alert = alert_data.iloc[-1]

    alert_level = str(
        alert["alert_level"]
    )

    alert_message = str(
        alert["alert_message"]
    )

    recommendation = str(
        alert["recommendation"]
    )

    if alert_level.upper() == "HIGH":

        st.error(
            f"🚨 {alert_message}"
        )

    elif alert_level.upper() == "MEDIUM":

        st.warning(
            f"⚠️ {alert_message}"
        )

    else:

        st.success(
            f"✅ {alert_message}"
        )

    st.markdown(
        "### 💡 Recommendation"
    )

    st.info(
        recommendation
    )

    # --------------------------------------------------------
    # Traffic insights
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Traffic Insights"
    )

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.write(
            f"🏎️ **Speed:** {alert['speed_insight']}"
        )

    with insight_col2:

        st.write(
            f"🚗 **Volume:** {alert['volume_insight']}"
        )

else:

    st.warning(
        "Traffic alert data is not available."
    )


# ============================================================
# TRAFFIC TREND ANALYSIS
# ============================================================

st.markdown("## 📈 Traffic Trend Analysis")


tab1, tab2, tab3 = st.tabs(
    [
        "🚗 Vehicle Count",
        "🏎️ Average Speed",
        "🚦 Congestion Score"
    ]
)


# ============================================================
# VEHICLE COUNT TREND
# ============================================================

with tab1:

    vehicle_chart = traffic_data[
        [
            "video_time_seconds",
            "vehicle_count"
        ]
    ].copy()

    vehicle_chart = vehicle_chart.set_index(
        "video_time_seconds"
    )

    st.line_chart(
        vehicle_chart
    )


# ============================================================
# SPEED TREND
# ============================================================

with tab2:

    speed_chart = traffic_data[
        [
            "video_time_seconds",
            "average_speed_kmh"
        ]
    ].copy()

    speed_chart = speed_chart.set_index(
        "video_time_seconds"
    )

    st.line_chart(
        speed_chart
    )


# ============================================================
# CONGESTION TREND
# ============================================================

with tab3:

    congestion_chart = traffic_data[
        [
            "video_time_seconds",
            "congestion_score"
        ]
    ].copy()

    congestion_chart = congestion_chart.set_index(
        "video_time_seconds"
    )

    st.line_chart(
        congestion_chart
    )


# ============================================================
# TRAFFIC DISTRIBUTION
# ============================================================

st.markdown("## 📊 Traffic Distribution")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Congestion Distribution
# ------------------------------------------------------------

with col1:

    st.markdown(
        "### 🚦 Congestion Levels"
    )

    congestion_counts = (
        traffic_data[
            "congestion_level"
        ]
        .value_counts()
    )

    st.bar_chart(
        congestion_counts
    )


# ------------------------------------------------------------
# Congestion Score Distribution
# ------------------------------------------------------------

with col2:

    st.markdown(
        "### 🎯 Congestion Score"
    )

    score_counts = (
        traffic_data[
            "congestion_score"
        ]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        score_counts
    )


# ============================================================
# RECENT TRAFFIC DATA
# ============================================================

st.markdown("## 📋 Recent Traffic Data")


recent_columns = [
    "video_time_seconds",
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "congestion_level"
]


recent_data = traffic_data[
    recent_columns
].tail(10)


st.dataframe(
    recent_data,
    width="stretch",
    hide_index=True
)


# ============================================================
# PROJECT STATISTICS
# ============================================================

st.markdown("---")

st.markdown(
    "## 📌 Project Information"
)


info1, info2, info3, info4 = st.columns(4)


with info1:

    duration = (
        traffic_data[
            "video_time_seconds"
        ].max()
        + 1
    )

    st.metric(
        "⏱️ Video Duration",
        f"{int(duration)} sec"
    )


with info2:

    st.metric(
        "📊 Data Points",
        len(traffic_data)
    )


with info3:

    st.metric(
        "🤖 Forecast Horizon",
        "5 sec"
    )


with info4:

    st.metric(
        "🎯 ML Model",
        "Random Forest"
    )


# ============================================================
# AI PIPELINE
# ============================================================

st.markdown("---")

st.markdown(
    "## 🧠 AI Traffic Intelligence Pipeline"
)

st.code(
    """
Traffic Video
      ↓
OpenCV
      ↓
YOLO Object Detection
      ↓
Vehicle Tracking
      ↓
Vehicle Counting
      ↓
ROI Analysis
      ↓
Traffic Density
      ↓
Speed Estimation
      ↓
Congestion Detection
      ↓
Historical Traffic Data
      ↓
Feature Engineering
      ↓
Machine Learning Forecasting
      ↓
Future Congestion Prediction
      ↓
Traffic Alert Engine
      ↓
Streamlit Dashboard
    """,
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Traffic Intelligence | YOLO + OpenCV + "
    "Machine Learning + Streamlit"
)