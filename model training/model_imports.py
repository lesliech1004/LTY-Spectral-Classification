import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("~/Documents/GitHub/LTY-Spectral-Classification/data files/lty_final.csv")

df_3bands = pd.read_csv("~/Documents/GitHub/LTY-Spectral-Classification/data files/lty_final_highacc.csv")

features = ["JH", "HK", "JK", "W1W2", "KW1", "KW2"]
y = df["spectral_type_code"]
X = df[features]

#stratify data so that the training and validation sets have a fair distribution of spectral types
y_bins = (y // 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y_bins)

#split our data into training and testing sets
y_3bands = df_3bands["spectral_type_code"]
X_3bands = df_3bands[features]

#stratify data so that the training and validation sets have a fair distribution of spectral types
y_3bands_bins = (y_3bands // 5)
X_3bands_train, X_3bands_test, y_3bands_train, y_3bands_test = train_test_split(X_3bands, y_3bands, test_size=0.2, random_state=42, stratify = y_3bands_bins)


def evaluate_model(model_name: str,
                   model: any,
                   y_pred: np.ndarray,
                   two_bands: bool,
                   colors_only: bool) -> None:
    if two_bands:
        y_test_use = y_test
        X_test_use = X_test
    else:
        y_test_use = y_3bands_test
        X_test_use = X_3bands_test

    mse = ((y_test_use - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    r2 = model.score(X_test_use, y_test_use)

    # Subtype-based metrics
    y_pred_rounded = np.rint(y_pred)  # round to nearest subtype code
    diff = np.abs(y_pred_rounded - y_test_use.values)

    within_1 = (diff <= 1).mean()   # within ±1 subtype
    within_2 = (diff <= 2).mean()   # within ±2 subtypes


    if two_bands:
        if colors_only:
            print(f"=== {model_name} results for samples with at least 2 bands trained on colors only ===")
        else:
            print(f"=== {model_name} results for samples with at least 2 bands ===")
    else:
        if colors_only:
            print(f"=== {model_name} results for samples with at least 3 bands trained on colors only ===")
        else:
            print(f"=== {model_name} results for samples with at least 3 bands ===")
            
    print(f"Mean Squared Error: {mse:.3f}") 
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"R^2 on test: {r2:.3f}") 
    print(f"Fraction within ±1 subtype: {within_1:.3f}") 
    print(f"Fraction within ±2 subtypes: {within_2:.3f}")