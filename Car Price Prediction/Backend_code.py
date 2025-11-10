# libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle  # Import the pickle library

# Load the dataset
# Add the encoding='latin1' parameter
df = pd.read_csv(r"C:\Users\DeLL\Desktop\Car_Price_Prediction\car_purchasing.csv", encoding='latin1')

# Drop unnecessary columns
df = df.drop(columns=['customer name', 'customer e-mail', 'country'])

# Split the dataset into dependent and independent variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --- Linear Regression Model ---
# 1. Create the model
lr_model = LinearRegression()

# 2. Fit on the original (unscaled) X_train
lr_model.fit(X_train, y_train)

# 3. Predict on the original (unscaled) X_test
y_pred_lr = lr_model.predict(X_test)

# 4. Get the R² score
score_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R² Score: {score_lr}")
print("Model trained successfully!")

# --- SAVE THE MODEL ---
# Now, save the trained model to a file using pickle
model_filename = 'car_price_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(lr_model, file)

print(f"Model saved to {model_filename}")