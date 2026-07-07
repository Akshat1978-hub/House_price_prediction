
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="House Price Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

df = load_data()

@st.cache_resource
def load_model():
    with open("house_model.pkl", "rb") as f:
        return pickle.load(f)

st.title("🏠 House Price Prediction Dashboard")

page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])

if page == "Dashboard":
    st.header("Analytics Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Houses", len(df))
    c2.metric("Average Price", f"{df['price'].mean():,.0f}")
    c3.metric("Max Price", f"{df['price'].max():,.0f}")
    c4.metric("Min Price", f"{df['price'].min():,.0f}")

    st.plotly_chart(px.histogram(df, x="price", title="Price Distribution"), use_container_width=True)

    if "area" in df.columns:
        st.plotly_chart(px.scatter(df, x="area", y="price", title="Area vs Price"), use_container_width=True)

    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index)
    )
    st.plotly_chart(fig, use_container_width=True)

    for col in ["bedrooms","bathrooms","stories","parking"]:
        if col in df.columns:
            st.plotly_chart(px.box(df, x=col, y="price", title=f"{col} vs Price"), use_container_width=True)

else:
    st.header("Predict House Price")

    model = load_model()

    inputs = {}
    for col in df.columns:
        if col == "price":
            continue

        if df[col].dtype == "object":
            inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
        else:
            inputs[col] = st.number_input(
                col,
                value=float(df[col].median())
            )

    if st.button("Predict"):
        sample = pd.DataFrame([inputs])
        pred = model.predict(sample)[0]
        st.success(f"Predicted House Price: ₹ {pred:,.2f}")
