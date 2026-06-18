# ============================================================
# TASK: Predict house prices using Linear Regression
# Features used: Square Footage, Bedrooms, Bathrooms
# ============================================================

# ---- Step 1: Import the libraries we need ----
import pandas as pd                                  # for loading & handling tabular data
from sklearn.model_selection import train_test_split # to split data into train/test
from sklearn.linear_model import LinearRegression    # our model
from sklearn.metrics import mean_squared_error, r2_score  # to evaluate the model
import matplotlib.pyplot as plt                       # for plotting

# ---- Step 2: Load the dataset ----
df = pd.read_csv("train.csv")
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# ---- Step 3: Select & engineer the features we need ----
# The task wants: square footage, bedrooms, bathrooms -> predict price

# GrLivArea = Above-ground living area in square feet (our "square footage")
# BedroomAbvGr = Number of bedrooms above ground
# FullBath = Number of full bathrooms
# HalfBath = Number of half bathrooms (counts as 0.5 of a full bathroom)

df["TotalBathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]

# Now build our feature table (X) and target column (y)
features = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]
X = df[features]          # inputs the model will learn from
y = df["SalePrice"]       # the value we want to predict

print("Sample of features (X):")
print(X.head())
print("\nSample of target (y):")
print(y.head())

# ---- Step 4: Split data into training and testing sets ----
# We train the model on 80% of the data, and test it on the remaining 20%
# it has NEVER seen, so we get an honest measure of performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {len(X_train)}, Testing rows: {len(X_test)}")

# ---- Step 5: Train (fit) the Linear Regression model ----
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained!")
print("Learned coefficients (weights):")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef:,.2f}")
print(f"Intercept (base price): {model.intercept_:,.2f}")

# ---- Step 6: Make predictions on the unseen test data ----
y_pred = model.predict(X_test)

# ---- Step 7: Evaluate how good the model is ----
rmse = mean_squared_error(y_test, y_pred) ** 0.5   # Root Mean Squared Error (avg $ error)
r2 = r2_score(y_test, y_pred)                       # R^2 score (0 to 1, higher = better fit)

print(f"\n--- Model Performance ---")
print(f"RMSE (average prediction error in $): {rmse:,.2f}")
print(f"R^2 Score (1.0 = perfect, 0 = no better than guessing average): {r2:.3f}")

# ---- Step 8: Visualize Actual vs Predicted prices ----
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
print("\nSaved plot as 'actual_vs_predicted.png'")

# ---- Step 9: Try predicting a new house's price ----
# Example: 1800 sq ft, 3 bedrooms, 2 bathrooms
example = pd.DataFrame({
    "GrLivArea": [1800],
    "BedroomAbvGr": [3],
    "TotalBathrooms": [2]
})
predicted_price = model.predict(example)[0]
print(f"\nExample prediction -> 1800 sqft, 3 bed, 2 bath house: ${predicted_price:,.2f}")
