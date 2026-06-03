import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Solar Energy Prediction",
    page_icon="☀️",
    layout="wide"
)

# ---------------------------------
# CSS
# ---------------------------------

st.markdown("""
<style>

.main{
    padding:1rem;
}

.kpi{
    background:#f8f9fa;
    padding:20px;
    border-radius:12px;
    border-left:5px solid orange;
    text-align:center;
}

.title{
    text-align:center;
    color:#ff6600;
    font-size:42px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD DATA
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "solar_model.pkl"
)

data_path = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "solar_data.csv"
)

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# ---------------------------------
# SIDEBAR
# ---------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "ℹ️ About Project",
        "📊 Dashboard",
        "📈 Analytics",
        "⚡ Prediction",
        "📑 Reports",
        "❓ Help",
    ]
)

# ---------------------------------
# HOME
# ---------------------------------

if page == "🏠 Home":

    st.markdown(
        '<p class="title">☀️ Solar Energy Prediction System</p>',
        unsafe_allow_html=True
    )

    st.write("### Machine Learning Based Solar Power Forecasting")

    st.info("""
    Predict solar energy generation using weather
    parameters and Machine Learning algorithms.
    """)

# ---------------------------------
# ABOUT
# ---------------------------------

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Solar Energy Prediction Project")

    st.markdown("---")

    # Project Introduction
    st.subheader("🌞 Project Introduction")

    st.write("""
    Solar energy is one of the fastest-growing renewable energy sources worldwide.
    Photovoltaic (PV) systems convert sunlight into electricity and play a vital role
    in sustainable energy production.

    However, solar power generation is highly dependent on weather conditions such as
    temperature, humidity, solar radiation, cloud cover, and wind speed. These factors
    cause fluctuations in power generation, making accurate forecasting essential for
    efficient energy management.

    This project uses Machine Learning techniques to predict solar energy generation
    based on meteorological parameters. The system analyzes historical weather data,
    identifies important patterns, and provides accurate solar power predictions
    through an interactive Streamlit dashboard.
    """)

    st.markdown("---")

    # Problem Statement
    st.subheader("⚠️ Problem Statement")

    st.write("""
    Solar power generation varies continuously due to changing weather conditions.
    Inaccurate prediction of solar energy can lead to power imbalance, inefficient
    energy storage, and financial losses in large-scale solar farms.

    Therefore, an intelligent Machine Learning-based system is required to forecast
    solar power generation accurately using weather-related parameters.
    """)

    st.markdown("---")

    # Objectives
    st.subheader("🎯 Project Objectives")

    st.markdown("""
    ✔ Predict hourly solar energy generation

    ✔ Analyze weather parameters affecting power production

    ✔ Improve energy planning and grid management

    ✔ Reduce uncertainty in renewable energy forecasting

    ✔ Develop an interactive Streamlit dashboard

    ✔ Support sustainable energy management
    """)

    st.markdown("---")

    # Technology Stack
    st.subheader("🛠 Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Programming & Development**
        - Python
        - Streamlit
        - GitHub
        """)

    with col2:
        st.info("""
        **Machine Learning & Analytics**
        - Scikit-Learn
        - Pandas
        - NumPy
        - Plotly
        """)

    st.markdown("---")

    # Key Features
    st.subheader("✨ Key Features")

    st.markdown("""
    🔹 Real-time Solar Energy Prediction

    🔹 Interactive Dashboard & Analytics

    🔹 Weather Parameter Analysis

    🔹 Machine Learning Forecasting Model

    🔹 Dynamic Visualizations

    🔹 Downloadable Reports

    🔹 User-Friendly Interface
    """)

    st.markdown("---")

    # Future Scope
    st.subheader("🚀 Future Scope")

    st.markdown("""
    - Integration with Live Weather APIs
    - Deep Learning Models (LSTM, ANN)
    - IoT-based Solar Monitoring
    - Cloud Deployment (AWS, Azure, GCP)
    - Smart Grid Integration
    - Mobile Application Development
    """)    

# ---------------------------------
# DASHBOARD
# ---------------------------------

elif page == "📊 Dashboard":

    st.header("Dashboard")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Records",
        len(df)
    )

    col2.metric(
        "Average Power",
        round(df["generated_power_kw"].mean(),2)
    )

    col3.metric(
        "Maximum Power",
        round(df["generated_power_kw"].max(),2)
    )

    col4.metric(
        "Minimum Power",
        round(df["generated_power_kw"].min(),2)
    )

    st.dataframe(df.head(20))

# ---------------------------------
# ANALYTICS
# ---------------------------------

elif page == "📈 Analytics":

    st.header("Analytics")

    fig = px.histogram(
        df,
        x="generated_power_kw",
        title="Solar Power Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    corr = df.corr(numeric_only=True)

    fig2 = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------------------------------
# PREDICTION
# ---------------------------------

elif page == "⚡ Prediction":

    st.header("Solar Power Prediction")

    col1,col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "Temperature",
            value=25.0
        )

        humidity = st.number_input(
            "Humidity",
            value=60.0
        )

        pressure = st.number_input(
            "Pressure",
            value=1013.0
        )

        cloud_cover = st.slider(
            "Cloud Cover",
            0,
            100,
            20
        )

        radiation = st.number_input(
            "Solar Radiation",
            value=500.0
        )

    with col2:

        wind_speed_10m = st.number_input(
            "Wind Speed 10m",
            value=5.0
        )

        wind_speed_80m = st.number_input(
            "Wind Speed 80m",
            value=7.0
        )

        wind_gust = st.number_input(
            "Wind Gust",
            value=10.0
        )

        zenith = st.number_input(
            "Zenith",
            value=45.0
        )

        azimuth = st.number_input(
            "Azimuth",
            value=180.0
        )

    if st.button("Predict Solar Power"):

        input_data = np.array([[

            temperature,
            humidity,
            pressure,
            0,
            0,
            cloud_cover,
            0,
            0,
            0,
            radiation,
            wind_speed_10m,
            0,
            wind_speed_80m,
            0,
            wind_speed_80m,
            0,
            wind_gust,
            0,
            zenith,
            azimuth

        ]])

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Solar Power = {prediction[0]:.2f} kW"
        )

        chart_df = pd.DataFrame({
            "Parameter":[
                "Temperature",
                "Humidity",
                "Pressure",
                "Radiation"
            ],
            "Value":[
                temperature,
                humidity,
                pressure,
                radiation
            ]
        })

        fig = px.bar(
            chart_df,
            x="Parameter",
            y="Value",
            title="Weather Parameters"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(f"""
        AI Insight

        Predicted Output:
        {prediction[0]:.2f} kW

        • High radiation improves power generation

        • High cloud cover reduces output

        • Current weather is favorable for solar generation
        """)

# ---------------------------------
# REPORTS
# ---------------------------------

elif page == "📑 Reports":

    st.header("Reports")

    st.dataframe(df)

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Dataset",
        csv,
        "solar_report.csv",
        "text/csv"
    )

# ---------------------------------
# HELP
# ---------------------------------

elif page == "❓ Help":

    st.header("User Guide")

    st.write("""
    1. Open Prediction Page

    2. Enter Weather Parameters

    3. Click Predict

    4. View Prediction & Analysis
    """)

