# Mercedes-Benz Vehicle Price Prediction: Three Powerful Models Comparison

This project compares three state-of-the-art gradient boosting models - **XGBoost**, **LightGBM**, and **CatBoost** - for predicting Mercedes-Benz vehicle prices, with LightGBM achieving the best performance at 96.00% R² accuracy.

## 🎯 Project Objective
To evaluate and compare the performance of three leading gradient boosting algorithms for predicting used Mercedes-Benz vehicle prices, determining the optimal model for automotive price estimation.

## 📊 Dataset Information
- **Source**: Used Car Dataset - Mercedes-Benz Vehicles
- **Size**: 13,119 vehicle records
- **Features**: 9 original attributes + 3 engineered features
- **Target Variable**: Vehicle price (regression problem)
- **Data Quality**: Clean dataset with no missing values

### Key Features
- **Vehicle Specs**: Model, year, mileage, engine size, fuel type
- **Performance Metrics**: MPG (miles per gallon), tax
- **Market Info**: Transmission type, price
- **Engineered Features**: Vehicle age, mileage per year, price per MPG

## 🔧 Technologies Used
```python
pandas==1.5.0
numpy==1.23.0
xgboost==1.7.0
lightgbm==3.3.0
catboost==1.1.0
scikit-learn==1.1.0
matplotlib==3.6.0
seaborn==0.12.0
```

## 🏆 Model Performance Comparison

| Model | MAE | RMSE | R² Score | Training Speed |
|-------|-----|------|----------|---------------|
| **LightGBM** 🥇 | 1,536 | 2,499 | **0.9600** | Fastest |
| **CatBoost** 🥈 | 1,553 | 2,523 | **0.9592** | Medium |
| **XGBoost** 🥉 | 1,552 | 2,706 | **0.9531** | Slowest |

### Winner: LightGBM
- **Best R² Score**: 96.00%
- **Lowest MAE**: £1,536
- **Fastest Training**: Optimal for production deployment

## 🔍 Feature Importance Analysis
Top predictive features identified across all models:

1. **MPG (Fuel Efficiency)** - Primary price determinant
2. **Mileage** - Vehicle usage indicator  
3. **Mileage per Year** - Usage intensity metric
4. **Model Type** - Mercedes model classification
5. **Engine Size** - Performance specification
6. **Vehicle Year** - Age/depreciation factor

## 🛠️ Data Preprocessing & Feature Engineering

### Original Features (9):
- Vehicle specifications and market attributes

### Engineered Features (3):
- **Vehicle Age**: `2024 - year`
- **Mileage per Year**: `mileage / (age + 1)`
- **Price per MPG**: `price / mpg`

### Encoding Strategy:
- **XGBoost & LightGBM**: Label encoding for categorical variables
- **CatBoost**: Native categorical feature handling (no preprocessing needed)

## 📈 Model Configuration

### XGBoost Parameters:
```python
XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```

### LightGBM Parameters:
```python
LGBMRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```

### CatBoost Parameters:
```python
CatBoostRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=50,
    cat_features=['model', 'transmission', 'fuelType']
)
```

## 🚀 Installation & Usage

### Requirements
```bash
pip install pandas numpy xgboost lightgbm catboost scikit-learn matplotlib seaborn
```

### Data Download
```bash
kaggle datasets download -d adityadesai13/used-car-dataset-ford-and-mercedes
```

### Execution
```bash
python mercedes.py
```

## 📊 Key Insights

### Model Performance Analysis:
1. **LightGBM Superiority**: Best overall performance with fastest training
2. **Close Competition**: All models achieved >95% R² accuracy
3. **CatBoost Efficiency**: Strong performance with minimal preprocessing
4. **XGBoost Reliability**: Solid baseline with extensive hyperparameter options

### Feature Insights:
1. **Fuel Efficiency Dominance**: MPG is the strongest price predictor
2. **Usage Patterns**: Mileage metrics significantly impact valuation
3. **Model Differentiation**: Mercedes model type strongly influences pricing
4. **Age Factor**: Vehicle depreciation captured through engineered features

## 💼 Business Applications
- **Automotive Pricing**: Dynamic pricing for used car dealerships
- **Insurance Valuation**: Vehicle worth assessment for claims
- **Fleet Management**: Asset depreciation tracking
- **Market Analysis**: Price trend identification and forecasting
- **Investment Decisions**: Vehicle portfolio optimization

## 🎯 Model Recommendations

### For Production Deployment:
**LightGBM** - Best choice due to:
- Highest accuracy (96.00% R²)
- Fastest training and inference
- Lower memory usage
- Excellent handling of categorical features

### For Experimentation:
**CatBoost** - Alternative option for:
- Automatic categorical handling
- Built-in regularization
- Robust overfitting protection

### For Baseline:
**XGBoost** - Traditional choice for:
- Extensive documentation
- Wide community support
- Proven track record

## ⚠️ Model Limitations
- Dataset limited to Mercedes-Benz vehicles only
- Price predictions based on UK market data
- Model performance may vary with different vehicle brands
- Temporal aspects not fully captured (market conditions)

## 🔮 Future Improvements
- **Multi-brand Analysis**: Extend to other luxury brands
- **Time Series Components**: Include market trend analysis
- **Ensemble Methods**: Combine all three models
- **Advanced Feature Engineering**: External data integration
- **Hyperparameter Optimization**: Grid/random search tuning

## 📈 Production Considerations
- **Model Monitoring**: Track prediction drift over time
- **Retraining Schedule**: Regular model updates with new data
- **A/B Testing**: Compare model versions in production
- **Confidence Intervals**: Provide prediction uncertainty

## 👤 Developer
- **Kaggle**: [@kurkigal](https://www.kaggle.com/kurkigal)

## 📄 License
MIT License

## 🤝 Contributing
Contributions welcome! Focus areas:
- Additional gradient boosting algorithms
- Advanced feature engineering techniques
- Cross-validation improvements
- Ensemble method implementations

---
⭐ If this comparison helps your automotive ML projects, please star the repository!