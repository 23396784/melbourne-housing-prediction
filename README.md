# Melbourne Housing Price Prediction 🏠

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project predicting residential property prices across three premium Melbourne suburbs using Random Forest, Decision Tree, and Linear Regression models. Developed as part of Deakin University's MSc Data Science & Business Analytics program (SIG720 Machine Learning).

## 📊 Project Highlights

| Metric | Best Model (Random Forest) |
|--------|---------------------------|
| **R² Score** | 0.679 (68% variance explained) |
| **MAE** | $236,643 |
| **RMSE** | $353,488 |
| **Dataset Size** | 150 properties |
| **Suburbs Analyzed** | Richmond, South Yarra, Hawthorn |

## 🎯 Problem Statement

The Melbourne real estate market is characterized by significant price variations across suburbs, property types, and features. This project develops and evaluates multiple regression models to predict housing prices, providing data-driven valuation tools for property buyers, sellers, and real estate professionals.

## 📁 Repository Structure

```
melbourne-housing-prediction/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── notebooks/
│   └── Melbourne_Housing_Prediction.ipynb  # Complete analysis notebook
├── reports/
│   └── Task_5D_Report.pdf              # Comprehensive project report
├── src/
│   ├── data_preprocessing.py           # Data cleaning and feature engineering
│   ├── model_training.py               # Model development and evaluation
│   └── gradio_app.py                   # Interactive web application
├── data/
│   └── README.md                       # Data description
└── images/
    └── feature_importance.png          # Visualization assets
```

## 🔬 Methodology

### Data Collection & Preprocessing
- **Manual data acquisition** from Melbourne real estate sources
- **150 property records** across Richmond, South Yarra, and Hawthorn
- Initial 15 features reduced to 6 key variables after quality assessment
- Features: Suburb, Property_Type, Bedrooms, Bathrooms, Car_Spaces, Distance_to_CBD

### Feature Engineering
- One-hot encoding for categorical variables (Suburb, Property_Type)
- Standardization of numerical features
- School access level feature creation based on proximity
- Temporal features (Sold_Year) for trend analysis

### Models Implemented
1. **Linear Regression** - Baseline model
2. **Decision Tree Regressor** - Non-linear relationships
3. **Random Forest Regressor** - Ensemble learning (best performer)
4. **XGBoost Regressor** - Gradient boosting comparison

### Evaluation Framework
- **K-Fold Cross-Validation** (5 folds)
- **Metrics**: MAE, RMSE, R² Score
- **SHAP Analysis** for model interpretability

## 📈 Results

### Model Comparison

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|-----|
| Linear Regression | 291,691 | 394,361 | 0.606 |
| Decision Tree | 237,073 | 358,550 | 0.668 |
| **Random Forest** | **236,643** | **353,488** | **0.679** |
| Tuned Random Forest | 459,829 | 1,191,285 | 0.682 |

*Note: Untuned Random Forest provides best balance across all metrics*

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Distance_to_CBD | ~80% |
| 2 | Bathrooms | ~20% |
| 3 | Bedrooms | ~0% (low variance in dataset) |

### Suburb Price Analysis

| Suburb | Avg Price | Distance to CBD | Characteristics |
|--------|-----------|-----------------|-----------------|
| Hawthorn | $355,746 | 7.5 km | Premium, family-oriented |
| South Yarra | $320,561 | 3.5 km | Location premium, inner-city |
| Richmond | $315,577 | 6.4 km | Value-focused, stable |

## 🖥️ Interactive Web Application

A Gradio-powered prediction interface was deployed with:
- Real-time price predictions
- Market comparison insights
- Suburb-specific analytics
- Confidence level indicators

```python
# Sample prediction
Property: 3BR/2BA Apartment in Hawthorn, 5km from CBD
Predicted Price: $382,791
Confidence Level: High
Market Comparison: 8.2% below similar properties
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/melbourne-housing-prediction.git
cd melbourne-housing-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/Melbourne_Housing_Prediction.ipynb
```

### Run the Web App

```bash
# Launch Gradio interface
python src/gradio_app.py
```

## 📊 Key Insights

### Market Findings
1. **Distance to CBD is the dominant price driver** (~80% importance)
2. **Hawthorn commands highest premiums** despite being furthest from CBD (lifestyle/space premium)
3. **Richmond offers best value** with lowest price-per-bedroom ratio
4. **South Yarra prices location over space** - minimal bedroom premium

### Investment Recommendations
- **Conservative investors**: Richmond (most stable, lowest volatility)
- **Growth-oriented**: Hawthorn (highest ceiling at $509K)
- **Location priority**: South Yarra (closest to CBD)

## 🛠️ Technologies Used

- **Python 3.10+**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning models
- **XGBoost** - Gradient boosting
- **SHAP** - Model interpretability
- **Matplotlib/Seaborn** - Visualization
- **Gradio** - Web application

## 📚 References

1. Breiman, L. (2001). "Random forests." *Machine Learning*, 45(1), 5-32.
2. Chen, T. & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *KDD '16*, 785-794.
3. Lundberg, S.M. & Lee, S.-I. (2017). "A unified approach to interpreting model predictions." *NeurIPS*, 4765-4774.
4. Malpezzi, S. (2003). "Hedonic pricing models: A selective and applied review." *Housing Economics and Public Policy*, 1(1), 67-89.

## 👤 Author

**Victor Prefa**
- Medical Doctor with 17+ years clinical experience
- MSc Data Science & Business Analytics, Deakin University
- Transitioning from healthcare to data science

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Deakin University SIG720 Machine Learning course
- Melbourne real estate data sources
- Open-source ML community

---

*This project demonstrates a complete machine learning workflow from data collection through model deployment, addressing practical needs for data-driven property valuation in the Melbourne real estate market.*
