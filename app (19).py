import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# set streamlit layout to wide
st.set_page_config(layout="wide")

# Load the  trained model
with open ('best_model.pkl','rb') as file:
  model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as scaler_file:
  scaler = pickle.load(scaler_file)

# Define the input features for the model, matching the X_train columns
Feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female',
                 'Gender_Male', 'HasCrCard_0', 'HasCrCard_1', 'IsActiveMember_0', 'IsActiveMember_1']

# Columns requiring scaling
scales_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Updated default values to match Feature_names (assuming common defaults for one-hot encoded features)
default_values = {
    'CreditScore': 600,   # CreditScore
    'Age': 30,    # Age
    'Tenure': 2,     # Tenure
    'Balance': 60000.0, # Balance
    'NumOfProducts': 1,     # NumOfProducts
    'EstimatedSalary': 50000.0, # EstimatedSalary
    'Geography_France': True,  # Geography_France
    'Geography_Germany': False, # Geography_Germany
    'Geography_Spain': False, # Geography_Spain
    'Gender_Female': True,  # Gender_Female
    'Gender_Male': False, # Gender_Male
    'HasCrCard_0': False, # HasCrCard_0 (assuming HasCrCard is 1 by default)
    'HasCrCard_1': True,  # HasCrCard_1
    'IsActiveMember_0': False, # IsActiveMember_0 (assuming IsActiveMember is 1 by default)
    'IsActiveMember_1': True   # IsActiveMember_1
}

# Sidebar setup
# st.sidebar.image('/content/openart-image_TEiYW09i_1768740498339_raw.png', width=100) # Removed due to FileNotFoundError
st.sidebar.header('User Inputs')

# Collect user inputs
user_inputs = {}
for feature in Feature_names:
    if feature == 'CreditScore':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=350, max_value=850, value=default_values[feature], step=1)
    elif feature == 'Age':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=18, max_value=92, value=default_values[feature], step=1)
    elif feature == 'Tenure':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=0, max_value=10, value=default_values[feature], step=1)
    elif feature == 'Balance':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=250000.0, value=default_values[feature], step=100.0)
    elif feature == 'NumOfProducts':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=1, max_value=4, value=default_values[feature], step=1)
    elif feature == 'EstimatedSalary':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=200000.0, value=default_values[feature], step=100.0)
    elif isinstance(default_values[feature], bool):
        user_inputs[feature] = st.sidebar.selectbox(feature, [0, 1], index=1 if default_values[feature] else 0)
    else:
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_values[feature])

# App header
# st.image('/content/openart-image_TEiYW09i_1768740498339_raw.png', width=700) # Removed due to FileNotFoundError
st.title('Customer Churn Prediction')

# Page layout
left_col, right_col = st.columns(2)

# Left page (Feature Importance)
with left_col:
  st.subheader('Feature Importance')
  # Load feature importance
  feature_importance = pd.read_excel('feature_importance.xlsx')
  # Plot the feature
  fig = px.bar(
    feature_importance.sort_values(by='Importance', ascending=False),
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature Importance'
  )
  st.plotly_chart(fig)

# Right page (Predictions)
with right_col:
  st.header('Predictions')
  if st.button('Predict'):
    # Convert user inputs to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Apply MinMaxScaler to the required columns
    input_df[scales_vars] = scaler.transform(input_df[scales_vars])

    # Make predictions
    prediction = model.predict(input_df)[0] # Get the first (and only) prediction
    probabilities = model.predict_proba(input_df)[:, 1] # Probability of class 1

    prediction_label = 'Churned' if prediction == 1 else 'Retain'

    # Display results
    st.subheader(f'Predicted Value: {prediction_label}')
    st.write(f'Predicted Probability (Churn): {probabilities[0]:.2f}')
    st.write(f'Predicted Probability (Retain): {1 - probabilities[0]:.2f}')

    # Display a clear output
    st.markdown(f"**## Output: {prediction_label}**")

!pip install pyngrok --quiet
from pyngrok import ngrok

!nohup streamlit run app.py &

# Kill any ngrok processes already running to free up the port
!kill $(lsof -t -i:8501)

# Authtoken setup, replace 'YOUR_NGROK_AUTHTOKEN' with your actual token if you have one
# from getpass import getpass
# authtoken = getpass("Enter your ngrok authtoken: ")
# ngrok.set_auth_token(authtoken)

# Connect to ngrok
public_url = ngrok.connect(addr='8501', proto='http', options={'bind_tls': True})
print(f"Streamlit App URL: {public_url}")
