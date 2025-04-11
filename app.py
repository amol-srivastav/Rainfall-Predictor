import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io

# Set title
st.title("Rainfall Prediction using LSTM")

# File uploader
uploaded_file = st.file_uploader("Upload your rainfall CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.head())

    if 'Rainfall' in data.columns:
        # Preprocess data
        data = data[data['Rainfall'].notna()]
        rainfall_data = data[['Rainfall']].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(rainfall_data)

        def create_sequences(data, time_steps=30):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps)])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)

        time_steps = 30
        X, y = create_sequences(data_scaled, time_steps)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Load pre-trained model
        model = load_model("rainfall_model.h5")

        # Train/test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Predict on test set
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluation Metrics
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        r2 = r2_score(y_actual, y_pred)

        st.subheader("Model Evaluation")
        st.markdown(f"**MSE:** {mse:.4f}")
        st.markdown(f"**MAE:** {mae:.4f}")
        st.markdown(f"**RMSE:** {rmse:.4f}")
        st.markdown(f"**MAPE:** {mape:.2f}%")
        st.markdown(f"**R² Score:** {r2:.4f}")

        # Plot actual vs predicted
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=y_actual.flatten(), name='Actual'))
        fig1.add_trace(go.Scatter(y=y_pred.flatten(), name='Predicted'))
        fig1.update_layout(title="Actual vs Predicted Rainfall", xaxis_title="Time", yaxis_title="Rainfall")
        st.plotly_chart(fig1)

        # Future prediction
        st.subheader("Future Forecast")
        future_days = st.slider("Select number of future days to predict", 1, 30, 10)

        recent_data = data_scaled[-time_steps:].reshape(1, time_steps, 1)
        future_predictions = []

        for _ in range(future_days):
            predicted_scaled = model.predict(recent_data)
            predicted_value = scaler.inverse_transform(predicted_scaled)
            future_predictions.append(predicted_value[0][0])

            predicted_scaled = predicted_scaled.reshape((1, 1, 1))
            recent_data = np.append(recent_data[:, 1:, :], predicted_scaled, axis=1)

        st.write("### Predicted Rainfall for Next Days")
        forecast_df = pd.DataFrame({"Day": list(range(1, future_days+1)), "Predicted Rainfall": future_predictions})
        st.write(forecast_df)

        # Download as CSV
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name='rainfall_forecast.csv',
            mime='text/csv')

        # Plot future prediction
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast_df['Day'], y=forecast_df['Predicted Rainfall'], name='Forecast'))
        fig2.update_layout(title=f"Predicted Rainfall for Next {future_days} Days", xaxis_title="Day", yaxis_title="Rainfall")
        st.plotly_chart(fig2)

        # Combined plot
        st.subheader("Combined Historical & Future Forecast")
        historical = scaler.inverse_transform(data_scaled[-100:]).flatten()
        combined = np.concatenate([historical, forecast_df['Predicted Rainfall'].values])

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=combined, name='Historical + Future'))
        fig3.update_layout(title="Historical + Predicted Rainfall", xaxis_title="Time Steps", yaxis_title="Rainfall")
        st.plotly_chart(fig3)

    else:
        st.error("'Rainfall' column not found in uploaded file. Please upload a correct CSV.")
