Task-01: House Price Prediction using Linear Regression

This project was completed as Task-01 of the Machine Learning Internship at Prodigy InfoTech.

📌 Task Objective

Implement a linear regression model to predict the price of a house based on:


Square footage (living area)
Number of bedrooms
Number of bathrooms


📂 Dataset

House Prices - Advanced Regression Techniques (Kaggle)
🔗 https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

The dataset contains 1,460 houses with 81 features each, including sale price. For this task, only the following relevant columns were used:

Feature UsedOriginal Column(s)DescriptionSquare FootageGrLivAreaAbove-ground living area (sq ft)BedroomsBedroomAbvGrNumber of bedrooms above groundBathroomsFullBath + HalfBathCombined into one value (half bath = 0.5)TargetSalePriceFinal sale price of the house ($)

🛠️ Approach


Loaded the dataset using pandas
Selected the 3 relevant features and engineered a combined "Total Bathrooms" column
Split the data into training (80%) and testing (20%) sets
Trained a Linear Regression model using scikit-learn
Evaluated performance using RMSE and R² score
Visualized Actual vs Predicted prices using matplotlib
Used the trained model to predict the price of a new sample house


📊 Results

MetricValueRMSE≈ $53,372R² Score≈ 0.629

Interpretation: The model explains about 63% of the variation in house prices using just 3 features. Predictions are generally accurate for mid-range homes, but tend to underestimate very high-priced homes — expected, since features like location, lot size, and home quality aren't included in this simple model.

Learned Model Equation

Price ≈ 100.64 × (Square Footage) − 26,645.53 × (Bedrooms) + 27,083.21 × (Bathrooms) + 56,862.58

Visualization

Show Image

💻 Tech Stack


Python
pandas
scikit-learn
matplotlib


▶️ How to Run


Clone this repository
Download train.csv from the Kaggle dataset link and place it in the same folder
Install dependencies:


   pip install pandas scikit-learn matplotlib


Run the script:


   python house_price_model.py

🔮 Future Improvements


Include more features (lot size, garage size, year built, neighborhood)
Try advanced models (Random Forest, Gradient Boosting, XGBoost) for better accuracy
Apply feature scaling and outlier removal for cleaner results



Internship: Prodigy InfoTech – Machine Learning Track
Task: 01 – Linear Regression
