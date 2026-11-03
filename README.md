# 🏡 California Housing Price Prediction: Project Overview & Roadmap

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-v2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()

> **Academic AI / Machine Learning Project (Option A — Kaggle Style):** A step-by-step regression project aimed at predicting median house values in California districts using Python and `scikit-learn`.

---

## 📌 Project Overview

Valuing real estate properties is a classic and challenging problem in predictive modeling. Property values depend on an intricate blend of geographic positioning, structural characteristics, and local socioeconomic factors.

This project focuses on building an end-to-end Machine Learning regression pipeline to estimate median house values across California census block groups.

### Objectives:
- Perform exploratory data analysis (EDA) to understand feature distributions and correlations.
- Implement and compare multiple regression models (e.g., standard baseline vs. non-linear ensemble methods).
- Evaluate model performances with standard statistical metrics ($MAE$, $MSE$, $RMSE$, and $R^2$).
- Deploy a clean inference function capable of predicting house prices for newly observed districts.

---

## 📊 Dataset: California Housing

The project will utilize the standard **California Housing Dataset** (derived from the 1990 U.S. Census), available through `sklearn.datasets.fetch_california_housing`.

- **Size:** $20,640$ district observations
- **Target:** `MedHouseVal` — Median house value for California districts (expressed in units of $\$100,000$)

### Features Dictionary

| Feature | Description | Unit / Scale |
| :--- | :--- | :--- |
| `MedInc` | Median income in block group | Tens of thousands of USD |
| `HouseAge` | Median house age in block group | Years |
| `AveRooms` | Average number of rooms per household | Count |
| `AveBedrms` | Average number of bedrooms per household | Count |
| `Population` | Total block group population | Count |
| `AveOccup` | Average number of household members | Ratio |
| `Latitude` | Geographic latitude of district center | Degrees North |
| `Longitude` | Geographic longitude of district center | Degrees West |

---

## 🗺️ Planned Methodology & Roadmap

The project is structured into 5 iterative phases inspired by Kaggle best practices:

```text
[Phase 1: Exploration]  ──>  [Phase 2: Preprocessing]  ──>  [Phase 3: Modeling]  ──>  [Phase 4: Evaluation]  ──>  [Phase 5: Inference]
```

### 1. Exploratory Data Analysis (EDA)
- Check data cleanliness and potential missing values.
- Compute summary statistics (mean, median, standard deviations).
- Analyze feature correlations with the target variable (`MedHouseVal`).

### 2. Preprocessing & Data Splitting
- Partition the dataset into **80% training** and **20% testing** partitions to prevent data snooping.
- Apply feature normalization (e.g., `StandardScaler`) where required using leak-free pipelines (`Pipeline`).

### 3. Model Training & Comparison
We plan to benchmark at least two distinct regression families:
- **Baseline Model:** Multiple Linear Regression (analytical OLS baseline).
- **Non-Linear Model:** Random Forest Regressor (ensemble of decision trees).

### 4. Quantitative Evaluation
The candidate models will be compared using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **$R^2$ Score (Coefficient of Determination)**

### 5. Single-Sample Inference
- Simulate custom property feature vectors (e.g., a candidate house in Northern California) to test real-world prediction output.

---

## 📁 Repository Structure

```text
HouseValue-ML/
│
├── .gitignore               # Standard Git ignore rules
├── LICENSE                  # MIT License
├── README.md                # Project documentation and roadmap
├── requirements.txt         # Required Python packages
│
├── train.py                 # (To be added) Main training & evaluation script
└── housing_prediction.ipynb # (To be added) Interactive step-by-step notebook
```

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Machine Learning:** `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Serialization:** `joblib`

---

## 🚀 Setup & Installation

To initialize the project environment locally:

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/HouseValue-ML.git
cd HouseValue-ML

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
