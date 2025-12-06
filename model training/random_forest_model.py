from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df = pd.read_csv("~/Documents/GitHub/LTY-Spectral-Classification/data files/lty_final.csv")

#split our data into training and testing sets
features = ["JH", "HK", "JK", "W1W2", "KW1", "KW2"]
y = df["spectral_type_code"]
X = df[features]

mask = X.notna().any(axis=1)
X = X[mask]
y = y[mask]

#stratify data so that the training and validation sets have a fair distribution of spectral types
y_bins = (y // 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y_bins)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
# Evaluate the model
mse = ((y_test - y_pred) ** 2).mean()
print(f"Mean Squared Error: {mse}") #4.5241287744987755
score = rf.score(X_test, y_test)
print("R^2 on test:", score) #0.892471718701953