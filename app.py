import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Food_Delivery_Times.csv"
MODEL_PATH = BASE_DIR / "best_rf_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

df = load_data()
model, label_encoders = load_artifacts()

# Exclude the identifier from all app-facing dataset analysis.
data = df.drop(columns=["Order_ID"], errors="ignore").copy()

st.title("🚴 Food Delivery Time Predictor")
st.caption("Explore delivery operations and estimate delivery time using your trained Random Forest model.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        ["Overview", "Predict Delivery Time", "Dataset Explorer", "Insights"]
    )
    st.divider()
    st.caption("Model: Random Forest Regressor")
    st.caption("Identifier excluded: Order_ID")

if page == "Overview":
    st.subheader("Delivery Operations Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{len(data):,}")
    c2.metric("Average Delivery Time", f"{data['Delivery_Time_min'].mean():.1f} min")
    c3.metric("Average Distance", f"{data['Distance_km'].mean():.1f} km")
    c4.metric("Average Preparation", f"{data['Preparation_Time_min'].mean():.1f} min")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Delivery time distribution")
        fig = px.histogram(
            data, x="Delivery_Time_min", nbins=25,
            labels={"Delivery_Time_min": "Delivery time (minutes)"},
            color_discrete_sequence=["#3B82F6"]
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Orders by traffic level")
        traffic = data["Traffic_Level"].value_counts().rename_axis("Traffic level").reset_index(name="Orders")
        fig = px.bar(traffic, x="Traffic level", y="Orders", color="Traffic level")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recent dataset records")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

elif page == "Predict Delivery Time":
    st.subheader("Predict a Delivery")
    st.write("Enter the delivery details below. The prediction uses the five features expected by your saved model.")

    weather_classes = list(label_encoders["Weather"].classes_)
    traffic_classes = list(label_encoders["Traffic_Level"].classes_)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            distance = st.number_input(
                "Distance (km)", min_value=0.1,
                max_value=float(max(50, data["Distance_km"].max() * 2)),
                value=float(data["Distance_km"].median()), step=0.1
            )
            weather = st.selectbox("Weather", weather_classes)
            traffic = st.selectbox("Traffic Level", traffic_classes)
        with col2:
            prep_time = st.number_input(
                "Preparation Time (minutes)", min_value=0, max_value=300,
                value=int(data["Preparation_Time_min"].median()), step=1
            )
            courier_exp = st.number_input(
                "Courier Experience (years)", min_value=0.0, max_value=50.0,
                value=float(data["Courier_Experience_yrs"].median()), step=0.5
            )
        submitted = st.form_submit_button("Predict Delivery Time", type="primary", use_container_width=True)

    if submitted:
        weather_encoded = int(label_encoders["Weather"].transform([weather])[0])
        traffic_encoded = int(label_encoders["Traffic_Level"].transform([traffic])[0])
        feature_names = list(getattr(model, "feature_names_in_", [
            "Distance_km", "Weather", "Traffic_Level",
            "Preparation_Time_min", "Courier_Experience_yrs"
        ]))
        values = {
            "Distance_km": distance,
            "Weather": weather_encoded,
            "Traffic_Level": traffic_encoded,
            "Preparation_Time_min": prep_time,
            "Courier_Experience_yrs": courier_exp
        }
        input_df = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
        prediction = float(model.predict(input_df)[0])

        st.success("Prediction generated successfully")
        a, b = st.columns(2)
        a.metric("Estimated Delivery Time", f"{prediction:.1f} minutes")
        a.metric("Estimated Time", f"{int(round(prediction))} minutes")
        b.metric("Estimated Delivery Time", f"{prediction / 60:.2f} hours")
        st.info("This estimate is generated by your saved Random Forest model. Actual delivery time may vary.")

        with st.expander("View model input"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    with st.expander("Why are Time_of_Day and Vehicle_Type not in the prediction form?"):
        st.write(
            "Those columns exist in your CSV, but the saved model was trained with only "
            "Distance_km, Weather, Traffic_Level, Preparation_Time_min, and "
            "Courier_Experience_yrs. They are therefore shown in the dataset explorer "
            "but are not passed to this model."
        )

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    st.write("All available dataset columns are included below, except `Order_ID`.")

    f1, f2 = st.columns(2)
    with f1:
        selected_weather = st.multiselect(
            "Filter by Weather",
            options=sorted(data["Weather"].dropna().unique()),
            default=sorted(data["Weather"].dropna().unique())
        )
    with f2:
        selected_traffic = st.multiselect(
            "Filter by Traffic Level",
            options=sorted(data["Traffic_Level"].dropna().unique()),
            default=sorted(data["Traffic_Level"].dropna().unique())
        )

    filtered = data[
        data["Weather"].isin(selected_weather) &
        data["Traffic_Level"].isin(selected_traffic)
    ]
    st.caption(f"Showing {len(filtered):,} of {len(data):,} records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="food_delivery_data_filtered.csv",
        mime="text/csv"
    )

elif page == "Insights":
    st.subheader("Delivery Performance Insights")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Average delivery time by weather")
        weather_avg = data.groupby("Weather", as_index=False)["Delivery_Time_min"].mean()
        fig = px.bar(weather_avg, x="Weather", y="Delivery_Time_min",
                     labels={"Delivery_Time_min": "Average delivery time (min)"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Distance vs delivery time")
        fig = px.scatter(data, x="Distance_km", y="Delivery_Time_min",
                         color="Traffic_Level", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Average delivery time by traffic")
        traffic_avg = data.groupby("Traffic_Level", as_index=False)["Delivery_Time_min"].mean()
        fig = px.bar(traffic_avg, x="Traffic_Level", y="Delivery_Time_min",
                     color="Traffic_Level",
                     labels={"Delivery_Time_min": "Average delivery time (min)"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Preparation time vs delivery time")
        fig = px.scatter(data, x="Preparation_Time_min", y="Delivery_Time_min",
                         color="Weather", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average delivery time by vehicle type")
    vehicle_avg = data.groupby("Vehicle_Type", as_index=False)["Delivery_Time_min"].mean()
    fig = px.bar(vehicle_avg, x="Vehicle_Type", y="Delivery_Time_min",
                 labels={"Delivery_Time_min": "Average delivery time (min)"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Food Delivery Analytics • Built with Streamlit")
