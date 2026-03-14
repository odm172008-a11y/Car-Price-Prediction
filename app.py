import pandas as pd 
import pickle as pk
import streamlit as st
import os

# Load the trained model from the same directory as this script.
# Place `model.pkl` next to `app.py` (or update the path below if you store it elsewhere).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pk.load(f)
else:
    st.error(f"model.pkl not found at: {MODEL_PATH}\nPlease place model.pkl next to app.py or update MODEL_PATH.")


def get_brand_name(car_name):
    """Extract only the brand (first word)"""
    return car_name.split(' ')[0].strip()

def get_model_name(car_name):
    """Extract only the model (everything after brand)"""
    parts = car_name.split(' ', 1)
    return parts[1].strip() if len(parts) > 1 else ""

def filter_brands(df):
    """Remove unwanted brands (for training/encoding only)."""
    unwanted_brands = ['Chevrolet', 'Datsun', 'Mitsubishi', 'Daewoo', 
                       'Fiat', 'Force', 'Ashok', 'Opel']
    return df[~df['brand'].isin(unwanted_brands)].reset_index(drop=True)

def encode_input(df):
    """Encode categorical variables to match training data"""
    df = df.copy()
   
    owner_mapping = {
        'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
        'Fourth & Above Owner': 4, 'Test Drive Car': 5
    }
    fuel_mapping = {
        'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4
    }
    seller_type_mapping = {
        'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3
    }
    transmission_mapping = {
        'Manual': 1, 'Automatic': 2
    }
    brand_mapping = {
        'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
        'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Jeep': 10,
        'Mercedes-Benz': 11, 'Audi': 12, 'Volkswagen': 13, 'BMW': 14,
        'Nissan': 15, 'Lexus': 16, 'Jaguar': 17, 'Land': 18, 'MG': 19,
        'Volvo': 20, 'Kia': 21, 'Ambassador': 22, 'Isuzu': 23
    }

    df['owner'] = df['owner'].map(owner_mapping).fillna(0)
    df['fuel'] = df['fuel'].map(fuel_mapping).fillna(0)
    df['seller_type'] = df['seller_type'].map(seller_type_mapping).fillna(0)
    df['transmission'] = df['transmission'].map(transmission_mapping).fillna(0)
    df['brand'] = df['brand'].map(brand_mapping).fillna(0)
    
    return df


cars_data = pd.read_csv('Cardetails.csv')

cars_data['brand'] = cars_data['name'].apply(get_brand_name)
cars_data['model'] = cars_data['name'].apply(get_model_name)


valid_brands = [
    'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Tata', 'Jeep', 'Mercedes-Benz', 'Audi', 'Volkswagen',
    'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Kia',
    'Ambassador', 'Isuzu'
]
cars_data_ui = cars_data[cars_data['brand'].isin(valid_brands)].reset_index(drop=True)


st.title("🚗 Car Price Prediction")
st.write("Fill in the details below to estimate your car's selling price.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox('Car Brand', sorted(valid_brands))
    model_name = st.selectbox('Car Model', sorted(cars_data_ui[cars_data_ui['brand'] == brand]['model'].unique()))
    year = st.number_input('Manufactured Year', min_value=1990, max_value=2024, value=2015)
    km_driven = st.number_input('Kms Driven', min_value=0, max_value=500000, value=10000)
    fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])

with col2:
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    mileage = st.number_input('Mileage (km/l)', min_value=5.0, max_value=50.0, value=18.0, step=0.1)
    engine = st.number_input('Engine CC', min_value=600, max_value=6000, value=1500)
    max_power = st.number_input('Max Power (bhp)', min_value=10, max_value=500, value=100)
    seats = st.number_input('Seats', min_value=2, max_value=10, value=5)


col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("🔍 Predict Price", use_container_width=True):

        feature_columns = ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        
       
        input_data_model = pd.DataFrame(
            [[brand, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=feature_columns
        )

        input_data_model = encode_input(input_data_model)

        if model is None:
            st.error("Error: Model not loaded. Please ensure `model.pkl` exists in the same folder as app.py.")
        elif input_data_model.isnull().values.any() or (input_data_model == 0).all().any():
            st.error("Error: Invalid input data. Please ensure all inputs match valid categories (e.g., valid car brand).")
        else:
            car_price = model.predict(input_data_model)
            st.success(f"💰 Estimated Car Price: ₹ {car_price[0]:,.2f}")