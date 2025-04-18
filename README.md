# Bengaluru House Price Prediction

A machine learning model that predicts real estate prices in Bengaluru with 95% accuracy, integrating data science techniques and web development for accessible property valuation.

## 📌 Project Overview

This project implements a house price prediction model for Bangalore, India, using machine learning techniques to provide accurate real estate valuations based on genuine factors rather than inflated market prices. The system:

- Predicts property prices using multiple regression algorithms with 95% accuracy
- Offers a user-friendly web interface built with Flask for accessible predictions
- Deploys a fully functional application on the Heroku Cloud Platform
- Provides valuable insights into the Bengaluru real estate market through data visualization

## 📊 Dataset

The dataset contains Bengaluru housing information with multiple features:
- Area type
- Location
- Size (BHK)
- Total square feet
- Bathrooms
- Balconies
- Availability status
- Society name

Source: [Kaggle - Bengaluru House Prices Dataset](https://www.kaggle.com/ameythakur20/bangalore-house-prices)

## 🔧 Tech Stack

- **Language**: Python
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Web Framework**: Flask
- **Deployment**: Heroku Cloud Platform

## 🧠 ML Models Used

- ✅ Decision Tree Regressor (Best Performing - 95% accuracy)
- Linear Regression
- Support Vector Regression
- Random Forest Regression
- Gradient Boosting Regressor

## 🧪 Data Preprocessing

- Comprehensive data cleaning and transformation
- Feature engineering and selection
- Dimensionality reduction techniques
- Outlier removal using:
  - Business logic
  - Standard deviation & mean methods
- Standardization with Standard Scaler

## 📈 Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Adjusted R² Score
- Root Mean Squared Log Error (RMSLE)

## 🔬 Hyper-Parameter Tuning Methods

- Halving Randomized Search CV (Recommended - fastest)
- Grid Search CV
- Randomized Search CV


## 🚀 Getting Started

Clone the repository:
```bash
git clone https://github.com/AjayVasan/Bangalore-House-Price-Prediction.git
cd Bangalore-House-Price-Prediction
```

Install required packages:
```bash
pip install -r requirements.txt
```

Run the Flask application:
```bash
python app.py
```

## 🔮 Future Scope

1. Experiment with different combinations of preprocessing techniques
2. Explore feature combinations and binning for improved accuracy
3. Implement additional regression methods like Elastic Net Regression
4. Enhance neighborhood data with environmental factors and crime rates

## 📚 References

1. [Bangalore House Price Prediction Model](https://www.kaggle.com/ameythakur20/bangalore-house-price-prediction-model)
2. [Heroku Documentation](https://devcenter.heroku.com/categories/reference)
3. [Web Application Repository](https://github.com/msatmod/Bangalore-House-Price-Prediction)
4. [Varma et al., "House Price Prediction Using Machine Learning and Neural Networks"](https://doi.org/10.1109/ICICCT.2018.8473231)
5. [Flask Web Development Documentation](https://flask.palletsprojects.com/)
