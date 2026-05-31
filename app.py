import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Sales Prediction System", page_icon="📈", layout="wide")

st.title("AI-Based Sales Prediction System")
st.markdown("Enter the details below to get a predicted sales value for any store and product family.")

@st.cache_resource
def load_model():
    model_path = "lgb_sales_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run the notebook first to train and save the model.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

FEATURE_COLS = [
    "store_nbr", "family_encoded", "store_type_encoded", "cluster", "city_encoded",
    "year", "month", "day_of_month", "day_of_week", "week_of_year", "is_weekend",
    "dcoilwtico", "is_holiday", "onpromotion", "has_promotion", "transactions",
    "sales_lag_16", "sales_lag_30", "sales_lag_60", "sales_lag_365",
]

FAMILY_MAP = {
    "AUTOMOTIVE": 0, "BABY CARE": 1, "BEAUTY": 2, "BEVERAGES": 3,
    "BOOKS": 4, "BREAD/BAKERY": 5, "CELEBRATION": 6, "CLEANING": 7,
    "DAIRY": 8, "DELI": 9, "EGGS": 10, "FROZEN FOODS": 11,
    "GROCERY I": 12, "GROCERY II": 13, "HARDWARE": 14, "HOME AND KITCHEN I": 15,
    "HOME AND KITCHEN II": 16, "HOME APPLIANCES": 17, "HOME CARE": 18,
    "LADIESWEAR": 19, "LAWN AND GARDEN": 20, "LINGERIE": 21,
    "LIQUOR,WINE,BEER": 22, "MAGAZINES": 23, "MEATS": 24,
    "PERSONAL CARE": 25, "PET SUPPLIES": 26, "PLAYERS AND ELECTRONICS": 27,
    "POULTRY": 28, "PREPARED FOODS": 29, "PRODUCE": 30,
    "SCHOOL AND OFFICE SUPPLIES": 31, "SEAFOOD": 32,
}

STORE_TYPE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

CITY_MAP = {
    "Ambato": 0, "Babahoyo": 1, "Cayambe": 2, "Cuenca": 3,
    "Daule": 4, "El Carmen": 5, "Esmeraldas": 6, "Guaranda": 7,
    "Guayaquil": 8, "Ibarra": 9, "Latacunga": 10, "Libertad": 11,
    "Loja": 12, "Machala": 13, "Manta": 14, "Playas": 15,
    "Puyo": 16, "Quevedo": 17, "Quito": 18, "Riobamba": 19,
    "Salinas": 20, "Santo Domingo": 21,
}

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Store Details")
    store_nbr = st.number_input("Store Number", min_value=1, max_value=54, value=1)
    store_type = st.selectbox("Store Type", list(STORE_TYPE_MAP.keys()))
    cluster = st.number_input("Store Cluster", min_value=1, max_value=17, value=1)
    city = st.selectbox("City", list(CITY_MAP.keys()))
    family = st.selectbox("Product Family", list(FAMILY_MAP.keys()))

with col2:
    st.subheader("Date Details")
    date_input = st.date_input("Prediction Date")
    is_holiday = st.selectbox("Is National Holiday?", ["No", "Yes"])

with col3:
    st.subheader("Sales Context")
    onpromotion = st.number_input("Items on Promotion", min_value=0, value=0)
    transactions = st.number_input("Expected Transactions", min_value=0, value=1000)
    oil_price = st.number_input("Oil Price (USD)", min_value=0.0, value=50.0, step=0.5)
    st.markdown("**Historical Sales (for lag features)**")
    lag_16  = st.number_input("Sales 16 days ago",  min_value=0.0, value=500.0)
    lag_30  = st.number_input("Sales 30 days ago",  min_value=0.0, value=500.0)
    lag_60  = st.number_input("Sales 60 days ago",  min_value=0.0, value=500.0)
    lag_365 = st.number_input("Sales 1 year ago",   min_value=0.0, value=500.0)

st.markdown("---")

if st.button("Predict Sales", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please check that lgb_sales_model.pkl is present.")
    else:
        d = pd.Timestamp(date_input)
        features = pd.DataFrame([{
            "store_nbr":          store_nbr,
            "family_encoded":     FAMILY_MAP[family],
            "store_type_encoded": STORE_TYPE_MAP[store_type],
            "cluster":            cluster,
            "city_encoded":       CITY_MAP[city],
            "year":               d.year,
            "month":              d.month,
            "day_of_month":       d.day,
            "day_of_week":        d.dayofweek,
            "week_of_year":       d.isocalendar()[1],
            "is_weekend":         int(d.dayofweek >= 5),
            "dcoilwtico":         oil_price,
            "is_holiday":         1 if is_holiday == "Yes" else 0,
            "onpromotion":        onpromotion,
            "has_promotion":      int(onpromotion > 0),
            "transactions":       transactions,
            "sales_lag_16":       lag_16,
            "sales_lag_30":       lag_30,
            "sales_lag_60":       lag_60,
            "sales_lag_365":      lag_365,
        }])

        prediction = float(np.clip(model.predict(features[FEATURE_COLS]), 0, None)[0])

        st.success(f"Predicted Sales: **{prediction:,.2f} units**")

        st.markdown("#### Input Summary")
        summary = {
            "Store": store_nbr, "Family": family, "Date": str(date_input),
            "City": city, "Store Type": store_type, "On Promotion": onpromotion,
            "Oil Price": oil_price, "Is Holiday": is_holiday,
        }
        st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))