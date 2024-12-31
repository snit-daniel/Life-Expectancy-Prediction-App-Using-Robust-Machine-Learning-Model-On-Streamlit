# 🌍 Life Expectancy Analysis and Prediction App

This Streamlit app provides an interactive way to explore life expectancy data, visualize key trends, and predict life expectancy based on specific features. It combines robust machine learning models with user-friendly visualization tools.

## 🚀 Features

- **Data Visualization**: 
  - Explore life expectancy trends across countries.
  - View a correlation heatmap of features influencing life expectancy.
  - Analyze the distribution of life expectancy data.

- **Prediction Tool**: 
  - Predict life expectancy based on factors such as `HIV/AIDS`, `Income composition of resources`, `Adult Mortality`, and more.

## 📁 Dataset
The app uses the `life_expectancy_data.csv` dataset. This dataset includes features like:
- `Life expectancy`
- `Year`
- `Status` (Developed or Developing)
- Health and socio-economic indicators such as `BMI`, `Schooling`, and `GDP`.

## 🛠️ How It Works

### 1. Data Visualization
- **Distribution of Life Expectancy**: Displays the distribution of life expectancy using histograms.
- **Trends by Country Status**: Compares life expectancy trends between developed and developing countries.
- **Correlation Heatmap**: Shows correlations between features to identify key influencers.

### 2. Prediction
- Input relevant features like `BMI` or `Schooling`.
- The model predicts life expectancy based on a trained stacking model (`stacking_model.pkl`).
- Data preprocessing is handled dynamically using `StandardScaler`.

## 🏗️ Model Details
The app uses a pre-trained machine learning model:
- **Type**: Stacking Regressor
- **Input Features**:
  - `HIV/AIDS`
  - `Income composition of resources`
  - `Adult Mortality`
  - `under-five deaths`
  - `Schooling`
  - `BMI`
  - `thinness 1-19 years`

The model was trained on cleaned data, with preprocessing pipelines for numerical and categorical variables.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/life-expectancy-app.git
   cd life-expectancy-app


2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Place the following files in the root directory:

  stacking_model.pkl
  life_expectancy_data.csv

4. Run the app:

 ```bash
streamlit run app.py
