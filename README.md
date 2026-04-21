# 🌍 Air Quality Index Prediction using Machine Learning

🚀 A machine learning project that predicts Air Quality Index (AQI) using pollutant data and provides an interactive Streamlit web interface.

---

## 📌 Overview

Air pollution is a major environmental and public health concern. This project focuses on predicting the **Air Quality Index (AQI) category** using machine learning techniques based on pollutant concentrations.

An interactive web application is developed using **Streamlit**, allowing users to select a location (state, city, station) and input pollutant values to obtain real-time AQI predictions.

---

## 🎯 Problem Statement

Traditional air quality monitoring systems provide raw pollutant data, which may not be easily interpretable by the general public. This project aims to:

* Simplify AQI understanding using machine learning
* Provide a user-friendly interface for prediction
* Enable location-based air quality insights

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Model:** Random Forest Classifier
* **Frontend:** Streamlit
* **Model Persistence:** Joblib

---

## 📊 Dataset Description

The dataset includes:

* 🌍 Location data: Country, State, City, Station
* 🧪 Pollutants: PM10, NH3, OZONE
* 📈 Measurements: Min, Max, Average pollutant values

### Data Preprocessing Steps:

* Handling missing values
* Transforming data from long format to wide format
* Feature selection for model training

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Cleaned missing values
* Converted dataset into structured format using pivot
* Selected relevant pollutant features

### 2. Model Training

* Algorithm: **Random Forest Classifier**
* Target: AQI Category
* Applied Label Encoding

### 3. Model Optimization

* Used **GridSearchCV** for hyperparameter tuning
* Selected best model based on cross-validation

### 4. Model Saving

* Saved trained model using `joblib`
* Saved label encoder for decoding predictions

---

## 🧠 How It Works

```
User Input (Location + Pollutants)
        ↓
Data Processing
        ↓
Trained ML Model (Random Forest)
        ↓
Prediction (Encoded Output)
        ↓
Label Decoding
        ↓
AQI Category (Color Display)
```

---

## 🌐 Application Features

* 📍 Location-based selection:

  * State → City → Station
* 🧪 User input for pollutant values
* 🎨 Color-coded AQI output:

  * 🟢 Good
  * 🟡 Satisfactory
  * 🟠 Moderate
  * 🔴 Poor
  * 🟣 Very Poor
  * ⚫ Severe
* ⚡ Real-time prediction

---

## 🖥️ Sample Output

**Input:**

```
PM10 = 120
NH3 = 25
OZONE = 60
```

**Output:**

```
AQI Category: Moderate (🟠)
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/jyothi-73/air-quality-prediction-ml.git
cd air-quality-prediction-ml
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Air-Quality-Prediction/
│
├── app.py                  # Streamlit application
├── aqi_rf_model.pkl        # Trained ML model
├── label_encoder.pkl       # Label encoder
├── main.ipynb              # Model training notebook
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## 📈 Results & Insights

* Successfully classified AQI categories based on pollutant levels
* Built an interactive and user-friendly web application
* Demonstrated practical use of machine learning in environmental analysis

---

## 🔮 Future Enhancements

* 🔗 Integrate real-time AQI APIs
* 📊 Add data visualization dashboards
* 🌐 Deploy application publicly
* 🤖 Improve model using advanced algorithms

---

## 💼 Resume Description

**Developed a Machine Learning-based Air Quality Index prediction system using Random Forest and deployed an interactive Streamlit web application with location-based filtering and real-time predictions.**

---

## 🎤 Viva / Interview Explanation

“This project predicts AQI category based on pollutant levels using a Random Forest model. I performed data preprocessing, feature selection, and hyperparameter tuning using GridSearchCV. Additionally, I developed a Streamlit interface for real-time, location-based AQI prediction.”

---

## 🙌 Conclusion

This project demonstrates how machine learning can be applied to environmental data to provide meaningful insights and user-friendly solutions for air quality monitoring.

---

## 👤 Author

**Jyothi**
B.Tech Data Science
Aspiring Data Analyst

---
