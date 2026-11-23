# 🏡 HouseValue-ML — California Housing Price Prediction

[![Course](https://img.shields.io/badge/Course-ITE%20351%3A%20AI%20%26%20Applications-1E3A8A)](https://gab1-flt.github.io/HouseValue-ML/blog)
[![Tech Blog](https://img.shields.io/badge/Tech%20Blog-Read%20Full%20Report-047857)](https://gab1-flt.github.io/HouseValue-ML/blog)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.7-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📖 **Full Technical Blog & Detailed Analysis:**
> All exploratory data analysis (EDA), spatial maps, correlation matrices, methodology details, and model diagnostics are available in our dedicated Tech Blog:
> 👉 **[Read the Full Tech Blog Online (GitHub Pages)](https://gab1-flt.github.io/HouseValue-ML/blog)** · **[View `blog.md` Source](blog.md)**

---

## 📌 Executive Summary

**HouseValue-ML** is a reproducible Machine Learning regression pipeline built with `scikit-learn` to predict median house values (`MedHouseVal`) and analyze real estate market trends across **`20,640` California census block groups** (`8` socioeconomic and geographic features, `0` missing values, `80% / 20%` train-test split, `random_state=42`).

---

## 📊 Key Results at a Glance

| Model | Test MAE (USD) | Test RMSE (USD) | Test R² | 5-Fold CV R² |
| :--- | ---: | ---: | ---: | ---: |
| **Linear Regression** *(Baseline)* | 53,320 USD | 74,558 USD | 0.5758 | 0.6115 ± 0.0124 |
| **Random Forest** *(200 trees)* | 32,659 USD | 50,391 USD | 0.8062 | 0.8058 ± 0.0063 |
| **HistGradientBoosting** *(Best)* | **29,454 USD** | **44,571 USD** | **0.8484** | **0.8400 ± 0.0046** |

- **Top Model:** `HistGradientBoostingRegressor` cuts test RMSE by **`40.2%`** (`29,987 USD`) compared to the linear baseline.
- **Primary Market Drivers (Permutation Importance):** `Latitude` (`0.9875`), `Longitude` (`0.8346`), and `MedInc` (`0.4145`).
- 👉 **[See all figures, feature statistics, and inference examples in the Full Tech Blog](https://gab1-flt.github.io/HouseValue-ML/blog)**

---

## 👥 Team Members (ITE 351 — ECE Paris)

| Member | Department & University | Email |
| :--- | :--- | :--- |
| **Corentin Lepeltier** | Computer Science & Engineering, ECE Paris (ITE 351) | `corentin.lepeltier@edu.ece.fr` |
| **Gabin Flotat** | Computer Science & Engineering, ECE Paris (ITE 351) | `gabin.flotat@edu.ece.fr` |
| **Roméo Poncelet** | Computer Science & Engineering, ECE Paris (ITE 351) | `romeo.poncelet@edu.ece.fr` |

---

## 📁 Repository Structure & Quick Start

```text
HouseValue-ML/
├── README.md                  # Concise project overview (this page)
├── blog.md                    # Full ITE 351 Tech Blog (Stage A report & analysis)
├── train.py                   # Reproducible ML training, CV, plotting & inference script
├── housing_prediction.ipynb   # Interactive Jupyter notebook walkthrough
├── requirements.txt           # Python dependencies
├── assets/figures/            # Generated EDA & evaluation figures (PNG)
└── results/metrics.json       # Exact descriptive statistics & benchmark metrics
```

```bash
git clone https://github.com/Gab1-Flt/HouseValue-ML.git
cd HouseValue-ML
pip install -r requirements.txt
python3 train.py
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
