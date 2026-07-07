import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

df = load_data()

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Dashboard", "Prediction"]
)

# ==========================
# DASHBOARD
# ==========================
if page == "Dashboard":

    st.title("🏠 House Price Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Houses",
        len(df)
    )

    col2.metric(
        "Average Price",
        f"₹ {df['price'].mean():,.0f}"
    )

    col3.metric(
        "Maximum Price",
        f"₹ {df['price'].max():,.0f}"
    )

    col4.metric(
        "Minimum Price",
        f"₹ {df['price'].min():,.0f}"
    )

    st.markdown("---")

    fig1 = px.histogram(
        df,
        x="price",
        nbins=30,
        title="Price Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="area",
        y="price",
        color="bedrooms",
        title="Area vs Price"
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "furnishingstatus" in df.columns:
        fig3 = px.box(
            df,
            x="furnishingstatus",
            y="price",
            title="Price by Furnishing Status"
        )
        st.plotly_chart(fig3, use_container_width=True)

    if "bedrooms" in df.columns:
        fig4 = px.bar(
            df.groupby("bedrooms")["price"].mean().reset_index(),
            x="bedrooms",
            y="price",
            title="Average Price by Bedrooms"
        )
        st.plotly_chart(fig4, use_container_width=True)

    if "bathrooms" in df.columns:
        fig5 = px.bar(
            df.groupby("bathrooms")["price"].mean().reset_index(),
            x="bathrooms",
            y="price",
            title="Average Price by Bathrooms"
        )
        st.plotly_chart(fig5, use_container_width=True)

# ==========================
# PREDICTION PAGE
# ==========================
else:

    st.title("🏠 Predict House Price")

    area = st.number_input("Area", min_value=500, value=5000)

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=10,
        value=2
    )

    mainroad = st.selectbox(
        "Main Road",
        ["yes", "no"]
    )

    guestroom = st.selectbox(
        "Guest Room",
        ["yes", "no"]
    )

    basement = st.selectbox(
        "Basement",
        ["yes", "no"]
    )

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"]
    )

    parking = st.number_input(
        "Parking",
        min_value=0,
        max_value=10,
        value=1
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"]
    )

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )

    if st.button("Predict Price"):

        input_df = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad": [mainroad],
            "guestroom": [guestroom],
            "basement": [basement],
            "hotwaterheating": [hotwaterheating],
            "airconditioning": [airconditioning],
            "parking": [parking],
            "prefarea": [prefarea],
            "furnishingstatus": [furnishingstatus]
        })

        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted House Price: ₹ {prediction:,.0f}"
        )

        st.balloons()
