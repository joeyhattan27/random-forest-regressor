import pandas as pd
import numpy as np
import joblib
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Life Expectancy Data.csv").drop(columns=["Unnamed: 0"])

# Clean up column names
clean_columns = []
for col in data.columns:
    col = col.strip()
    clean_columns.append(col)
data.columns = clean_columns

# Drop rows with no life expectancy
data = data[~data["Life expectancy"].isna()]

# Split data
X = data.drop(columns =["Life expectancy"])
y = data["Life expectancy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fix missing values
categorical_cols = ["Country", "Status"]
numerical_cols = list(X_train.drop(columns= categorical_cols).columns)

# For numerical
num_imputer = SimpleImputer(strategy="mean")
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# For categorical
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Encode categorical variables
encoder = OneHotEncoder(sparse_output = False, handle_unknown="ignore")
encoder.set_output(transform="pandas")
encoded_df_train = encoder.fit_transform(X_train[categorical_cols])
X_train = X_train.drop(columns=categorical_cols).join(encoded_df_train)
encoded_df_test = encoder.transform(X_test[categorical_cols])
X_test = X_test.drop(columns = categorical_cols).join(encoded_df_test)

# Scale all variables to normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and fun model
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate model
print(f"Predicted life expectancies are {predictions[:20]}")
print(f"Actual life expectancies are: {y_test.to_numpy()[:20]}")
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")

