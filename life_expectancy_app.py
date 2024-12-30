import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Load the trained model
model = joblib.load('stacking_model.pkl')

# Define the feature names used in the prediction model
prediction_features = [
    'HIV/AIDS',
    'Income composition of resources',
    'Adult Mortality',
    'under-five deaths',
    'Schooling',
    'BMI',
    'thinness 1-19 years'
]



# Set up page configuration
st.set_page_config(page_title="Life Expectancy Analysis", page_icon="🌍", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "Prediction"])

# --- Home Page ---
if page == "Home":
    st.title("🌍 Welcome to the Life Expectancy Analysis App")
    st.markdown(
        """
        Life expectancy is a key indicator of a nation's health and development. 
        This app allows you to:
        - **Visualize life expectancy** across different countries.
        - **Predict life expectancy** based on specific factors.
        
        Explore the data and predictions to understand the factors that contribute to better health and longer lives!
        """
    )
    st.image("https://i.imgur.com/pkENtpx.png", caption="Life Expectancy Analysis", use_container_width=True)

# --- Data Visualization Page ---
elif page == "Data Visualization":
    st.title("📊 Data Visualization: Life Expectancy by Country")
    st.markdown("Explore life expectancy trends across countries.")

    # Load and preprocess your data (assuming df is loaded correctly)
    df = pd.read_csv('life_expectancy_data.csv')
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

    # Visualization 1: Distribution of Life Expectancy
    st.subheader("Distribution of Life Expectancy")
    if 'Life expectancy' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Life expectancy'], bins=20, kde=True, color='blue', ax=ax)
        ax.set_title("Distribution of Life Expectancy", fontsize=16)
        ax.set_xlabel("Life Expectancy", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        st.pyplot(fig)
    else:
        st.error("The column 'Life expectancy' is not found in the dataset.")

    # Visualization 2: Life Expectancy Trends by Year and Status
    st.subheader("Life Expectancy Trends: Developed vs Developing Countries")
    data1 = df[['Year', 'Status', 'Life expectancy']]

    # Handle missing values
    data1 = data1.dropna(subset=['Life expectancy'])

    # Group by Year and Status to calculate the mean life expectancy
    status_avg = data1.groupby(['Year', 'Status'])['Life expectancy'].mean().unstack()

    # Plot life expectancy by Status
    fig, ax = plt.subplots(figsize=(12, 6))
    status_avg.plot(ax=ax, marker='o')
    ax.set_title('Life Expectancy Trends: Developed vs Developing Countries', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Life Expectancy', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(title='Status', fontsize=12)
    st.pyplot(fig)

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap of Features")

        # Ensure that the columns exist in the DataFrame before selecting them
    heatmap_columns = [
        'Year', 'Life expectancy', 'Adult Mortality', 'infant deaths', 'Alcohol',
        'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 'under-five deaths',
        'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
        'thinness 1-19 years', 'thinness 5-9 years', 'Income composition of resources', 'Schooling'
    ]

    # Only keep columns that exist in the DataFrame
    existing_columns = [col for col in heatmap_columns if col in df.columns]
    df = df[existing_columns]

    # Drop rows with missing values to avoid issues in correlation calculation
    df = df.dropna()

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Features', fontsize=16)
    st.pyplot(fig)


# --- Prediction Page ---
elif page == "Prediction":
    st.title("🔮 Life Expectancy Prediction")
    st.markdown(
        """
        Enter the values for the features below to predict life expectancy.
        """
    )

    # Adjust column layout for input fields
    col1, col2, col3 = st.columns(3)
    inputs = {}

    # Create input fields for each feature
    for i, feature in enumerate(prediction_features):
        col = [col1, col2, col3][i % 3]  # Distribute inputs across columns
        inputs[feature] = col.number_input(feature, min_value=0.0, max_value=100.0, step=0.1)

    # Convert inputs to numpy array
    input_data = np.array(list(inputs.values())).reshape(1, -1)

    # Apply StandardScaler to the input data
    scaler = StandardScaler()

    # Fit the scaler on the training data (assuming your model was trained on scaled data)
    # You can also load the scaler from a file if it was saved during model training
    # scaler = joblib.load('scaler.pkl')  # Uncomment if you have a saved scaler

    input_data_scaled = scaler.fit_transform(input_data)

    # Predict button
    if st.button("Predict Life Expectancy"):
        prediction = model.predict(input_data_scaled)
        st.success(f"Predicted Life Expectancy: **{prediction[0]:.2f} years**")
