import streamlit as st
import pickle
import numpy as np

# Load the model and the dataframe
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
except Exception as e:
    st.error(f"Error loading dataframe: {e}")

# Check if both files are loaded successfully
if 'pipe' in locals() and 'df' in locals():
    st.title("Laptop Price Predictor")

    # Brand
    company = st.selectbox('Brand', df['Company'].unique())

    # Type of laptop
    type = st.selectbox('Type', df['TypeName'].unique())

    # Ram
    ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

    # Weight
    weight = st.number_input('Weight (in kg)')

    # Touchscreen
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

    # Ips
    ips = st.selectbox('Ips', ['No', 'Yes'])

    # Screen size
    screen_size = st.number_input('Screen Size (in inches)')

    # Resolution
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])

    # CPU
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

    # HDD
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1000, 2000])

    # SSD
    ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1000])

    # GPU
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())

    # OS
    os = st.selectbox('OS', df['os'].unique())

    if st.button('Predict Price'):
        # Query
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        x_res = int(resolution.split('x')[0])
        y_res = int(resolution.split('x')[1])
        ppi = np.sqrt((x_res**2) + (y_res**2))/screen_size

        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, -1)

        try:
            # Predict the price
            prediction = str(int(np.exp(pipe.predict(query))))
            st.title(f"The predicted price of the laptop is: ₹{prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("Model or DataFrame not loaded properly.")

