# 📊 Mutual Fund NAV Prediction & Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-6_Models-success?style=for-the-badge)
![Time Series](https://img.shields.io/badge/Time_Series-Forecasting-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### Intelligent Mutual Fund Analysis & NAV Prediction using Machine Learning and Time Series Forecasting

**🔗 Live Application:** https://mutualfundanalysis-app.streamlit.app/

</div>

---

# 📖 Overview

Mutual Fund NAV Prediction & Analysis is an interactive **Streamlit-based Machine Learning application** that enables users to analyze historical Net Asset Value (NAV) trends of Indian mutual funds and forecast future NAV values using multiple forecasting techniques.

The application retrieves mutual fund data through the **AMFI (Association of Mutual Funds in India) API**, performs historical trend analysis, visualizes fund performance, and generates future NAV predictions using traditional Machine Learning, Statistical Time Series, and Deep Learning models.

The objective of this project is to demonstrate how multiple forecasting algorithms can be applied to financial time series data and compared within an interactive dashboard to support better investment analysis.

---

# 🚀 Live Demo

### 🌐 Streamlit Application

https://mutualfundanalysis-app.streamlit.app/

---

# ✨ Key Features

## 📈 Historical NAV Analysis

- Analyze historical NAV trends of Indian mutual funds
- Interactive performance visualization
- Time-series trend exploration
- Historical growth analysis
- NAV comparison across different periods

---

## 🤖 Machine Learning & Forecasting Models

The application supports multiple forecasting algorithms for NAV prediction:

| Model | Category |
|--------|----------|
| Linear Regression | Machine Learning |
| Random Forest Regression | Ensemble Learning |
| ARIMA | Statistical Forecasting |
| Auto Regression | Time Series Analysis |
| Simple Moving Average (SMA) | Trend Analysis |
| LSTM Neural Network | Deep Learning |

Users can compare different forecasting approaches to better understand prediction behavior and model performance.

---

## 📊 Interactive Dashboard

- Dynamic Streamlit Interface
- Interactive Charts
- Historical NAV Visualization
- Future NAV Prediction
- Model Comparison
- Easy Mutual Fund Selection
- Responsive User Interface

---

## 📉 Financial Analysis

- Historical NAV Performance
- Trend Identification
- Price Movement Analysis
- Forecast Visualization
- Comparative Analysis
- Investment Insights

---

# 🏗️ Project Architecture

```
                 AMFI API
                     │
                     ▼
        Historical Mutual Fund NAV Data
                     │
                     ▼
        Data Collection & Preprocessing
                     │
                     ▼
          Feature Preparation
                     │
                     ▼
      ┌─────────────────────────────┐
      │ Prediction Models           │
      │                             │
      │ • Linear Regression         │
      │ • Random Forest             │
      │ • ARIMA                     │
      │ • Auto Regression           │
      │ • SMA                       │
      │ • LSTM                      │
      └─────────────────────────────┘
                     │
                     ▼
         Forecast Future NAV Values
                     │
                     ▼
        Interactive Streamlit Dashboard
```

---

# ⚙️ Technology Stack

## Programming Language

- Python

## Web Framework

- Streamlit

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn

## Deep Learning

- TensorFlow
- Keras
- LSTM

## Time Series Forecasting

- Statsmodels
- ARIMA
- Auto Regression

## Data Visualization

- Plotly
- Matplotlib

## API Integration

- AMFI API (Association of Mutual Funds in India)

---

# 📂 Repository Structure

```
MUTUAL_FUND_ANALYSIS
│
├── .devcontainer/
├── app.py
├── requirements.txt
├── scheme_codes.json
├── __init__.py
├── .gitignore
└── README.md
```

---

# 📡 Data Source

The project uses the **AMFI (Association of Mutual Funds in India) API** to retrieve real mutual fund scheme information and historical NAV data.

The fetched data is processed and used for:

- Historical NAV Analysis
- Time Series Visualization
- Machine Learning Training
- Future NAV Forecasting

---

# 📊 Machine Learning Workflow

```
AMFI API
      │
      ▼
Historical NAV Data
      │
      ▼
Data Cleaning
      │
      ▼
Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
NAV Prediction
      │
      ▼
Interactive Dashboard
```

---

# 📚 Forecasting Models

### 📉 Linear Regression

A baseline machine learning model used for identifying linear trends in historical NAV data.

---

### 🌳 Random Forest Regression

An ensemble learning algorithm capable of modeling complex and non-linear relationships.

---

### 📊 ARIMA

A statistical time-series forecasting model widely used for financial market prediction.

---

### 📈 Auto Regression

Forecasts future NAV values using previous historical observations.

---

### 📉 Simple Moving Average (SMA)

Smooths short-term fluctuations to highlight long-term trends.

---

### 🧠 Long Short-Term Memory (LSTM)

A Deep Learning recurrent neural network capable of learning sequential financial patterns for long-term forecasting.

---

# 📸 Application Screenshots


Example:
**dashboard.png**
<img width="1897" height="902" alt="image" src="https://github.com/user-attachments/assets/5fcc3578-793a-475c-8980-06c8ec7b97e0" />

**prediction.png**
<img width="1897" height="897" alt="image" src="https://github.com/user-attachments/assets/768742ad-19b4-431a-ada9-2406f77b83c5" />

**comparison.png**
<img width="1897" height="822" alt="image" src="https://github.com/user-attachments/assets/b769e9bc-69cf-455a-8703-8bbbe6c851ca" />
<img width="1912" height="897" alt="image" src="https://github.com/user-attachments/assets/979749b7-189f-48fb-963b-fb0b4ec0ca95" />

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/TanishMhatre124/MUTUAL_FUND_ANALYSIS.git
```

Move into the project

```bash
cd MUTUAL_FUND_ANALYSIS
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 💡 Learning Outcomes

This project demonstrates practical implementation of:

- Financial Data Analysis
- Machine Learning
- Deep Learning
- Time Series Forecasting
- API Integration
- Data Visualization
- Interactive Dashboard Development
- End-to-End Data Science Workflow

---

# 🛠 Skills Demonstrated

- Python Programming
- Machine Learning
- Time Series Forecasting
- Financial Analytics
- Data Cleaning
- Feature Engineering
- Model Evaluation
- Deep Learning (LSTM)
- Streamlit
- API Integration
- Plotly Visualization

---

# 🚀 Future Enhancements

- Portfolio Recommendation System
- Risk Analysis Dashboard
- CAGR Calculator
- SIP Calculator
- Sharpe Ratio Analysis
- Live Market News Integration
- Prophet Forecasting
- XGBoost Forecasting
- Model Performance Comparison
- Export Prediction Reports
- User Authentication

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## **Tanish Mhatre**

**Data Analyst | Machine Learning Enthusiast | Python Developer**

### GitHub

https://github.com/TanishMhatre124

### Live Demo

https://mutualfundanalysis-app.streamlit.app/

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

Building data-driven solutions for smarter financial decision-making.

</div>
