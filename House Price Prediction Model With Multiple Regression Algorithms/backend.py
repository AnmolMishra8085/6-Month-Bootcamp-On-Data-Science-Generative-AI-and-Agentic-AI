# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\DeLL\Python Data Science Work\New Project\USA_Housing.csv")

# Preprocessing
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define models with names matching Flask app
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),  # Match Flask name “SVM”
    'LGBM': LGBMRegressor(),
    'XGBoost': XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

# Train and evaluate models
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })

    # Save model with consistent name
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print('✅ All models trained and saved successfully.')
print('📊 Evaluation results saved to model_evaluation_results.csv.')
