from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd

df = pd.read_csv("~/Documents/GitHub/LTY-Spectral-Classification/data files/lty_final.csv")

xgb = XGBRegressor(n_estimators=500,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

#split our data into training and testing sets
features = ["JH", "HK", "JK", "W1W2", "KW1", "KW2"]
y = df["spectral_type_code"]
X = df[features]

#stratify data so that the training and validation sets have a fair distribution of spectral types
y_bins = (y // 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y_bins)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Evaluate the model
mse = ((y_test - y_pred) ** 2).mean()
print(f"Mean Squared Error: {mse}") #3.7164334585889263
score = xgb.score(X_test, y_test)
print("R^2 on test:", score) #0.9116688046960163
