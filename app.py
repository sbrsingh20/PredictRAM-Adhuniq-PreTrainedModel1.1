import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Streamlit App Configuration
st.title("Stock Return Prediction App")
st.markdown("""
This app allows users to:
- Upload a pre-trained model.
- View the model's parameters and accuracy.
- Select multiple stocks for prediction.
- Input macroeconomic parameters to simulate a scenario (Inflation, Interest Rate, and VIX).
- View historical stock performance and predicted returns.
""")

# Step 1: Upload Pre-Trained Model
model_file = st.file_uploader("Upload Pre-Trained Model (.pkl file)", type=["pkl"])

if model_file:
    try:
        # Load the pickle file
        model_details = joblib.load(model_file)

        # Debugging: Display loaded object structure
        st.write("Loaded Object Type:", type(model_details))

        # Handle different model structures
        if isinstance(model_details, dict):
            model = model_details.get('model')
            model_params = model_details.get('parameters', {})
            model_accuracy = model_details.get('accuracy', 'Not available')
        elif hasattr(model_details, 'predict'):  # If directly a model
            model = model_details
            model_params = model.get_params() if hasattr(model, 'get_params') else "No parameters found"
            model_accuracy = "Not available"
        else:
            model = None
            model_params = {}
            model_accuracy = "Unknown"

        if model:
            st.success("Model loaded successfully!")
            st.subheader("Model Details")
            st.write("**Model Type:**", type(model).__name__)
            st.write("**Model Parameters:**", model_params)
            st.write("**Model Accuracy:**", model_accuracy)

            # Step 2: Select Stocks
            stock_files = [f for f in os.listdir('stockdata') if f.endswith('.xlsx')]
            available_stocks = [f.split('.')[0] for f in stock_files]
            selected_stocks = st.multiselect("Select Stocks", available_stocks)

            if selected_stocks:
                # Date Selection
                start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
                end_date = st.date_input("End Date", value=pd.to_datetime('2024-09-27'))

                # Step 3: Load Macroeconomic Data
                gdp_data = pd.read_excel('GDP data.xlsx', sheet_name='Sheet1', parse_dates=['Date'])
                st.subheader("Macroeconomic Data Overview")
                st.write(gdp_data.head())  # Display first few rows

                # Step 4: Input Macroeconomic Scenario
                inflation_rate = st.number_input("Inflation Rate (%)", value=gdp_data['Inflation'].mean(), format="%.2f")
                interest_rate = st.number_input("Interest Rate (%)", value=gdp_data['Interest Rate'].mean(), format="%.2f")
                vix_value = st.slider("VIX", 0.0, 100.0, gdp_data['VIX'].mean(), step=0.1)

                # Prepare New Scenario for Prediction
                new_scenario = np.array([inflation_rate, interest_rate, vix_value]).reshape(1, -1)

                # Step 5: Display Historical Stock Performance
                st.subheader("Historical Stock Performance")
                for stock in selected_stocks:
                    stock_file = f"stockdata/{stock}.xlsx"
                    if os.path.exists(stock_file):
                        stock_data = pd.read_excel(stock_file, parse_dates=['Date'], index_col='Date')
                        stock_data = stock_data[(stock_data.index >= pd.to_datetime(start_date)) &
                                                (stock_data.index <= pd.to_datetime(end_date))]

                        if not stock_data.empty:
                            # Plot Historical Data
                            fig, ax = plt.subplots()
                            stock_data['Close'].plot(ax=ax, title=f"Historical Prices for {stock}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Price (Close)")
                            st.pyplot(fig)
                        else:
                            st.warning(f"No data available for {stock} in the selected date range.")
                    else:
                        st.warning(f"No data file found for {stock}.")

                # Step 6: Predict Returns
                if st.button("Predict Returns"):
                    st.subheader("Predicted Returns for Selected Stocks")
                    for stock in selected_stocks:
                        try:
                            predicted_return = model.predict(new_scenario)[0]
                            st.write(f"**{stock}**: {predicted_return:.2f}%")

                            # Plot Predicted Return
                            fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

                            # Fetch and Plot Stock Data Again
                            stock_file = f"stockdata/{stock}.xlsx"
                            if os.path.exists(stock_file):
                                stock_data = pd.read_excel(stock_file, parse_dates=['Date'], index_col='Date')
                                stock_data = stock_data[(stock_data.index >= pd.to_datetime(start_date)) &
                                                        (stock_data.index <= pd.to_datetime(end_date))]
                                if not stock_data.empty:
                                    stock_data['Close'].plot(ax=ax[0], color="blue", title=f"Historical Prices for {stock}")
                                    ax[0].set_ylabel("Price (Close)")
                                    ax[0].set_xlabel("")

                            ax[1].bar(['Predicted Return'], [predicted_return], color='orange')
                            ax[1].set_ylabel('Return (%)')
                            ax[1].set_title(f'Predicted Return for {stock}')

                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Prediction error for {stock}: {e}")
            else:
                st.warning("Please select at least one stock to proceed.")
        else:
            st.error("No valid model found in the uploaded file.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("Please upload a valid pre-trained model file to proceed.")
