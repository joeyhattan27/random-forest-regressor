import pandas as pd
import numpy as np
import joblib
import matplotlib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Life Expectancy Data.csv").drop(columns=["Unnamed: 0"])

# Clean up column names
clean_columns = []
for col in data.columns:
    col = col.strip()
    clean_columns.append(col)
data.columns = clean_columns

# Drop rows with no life expectancy
data = data.dropna(subset=["Life expectancy"]).copy()

# Split data
X = data.drop(columns=["Life expectancy"])
y = data["Life expectancy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Split columns
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

# Build preprocessor for pipeline
num_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

# Build pipeline with model
pipeline = Pipeline(
    [("preprocessor", preprocessor), ("model", RandomForestRegressor())]
)


# Build cross validator
cv = KFold(5, shuffle=True, random_state=1)

baseline_scores = cross_val_score(
    estimator=pipeline, X=X_train, y=y_train, cv=cv, scoring="r2"
)
print(f"Baseline model r-sqaure score: {baseline_scores.mean()}")

param_dist = {
    "model__n_estimators": [25, 50, 100, 200, 400],
    "model__max_features": [1.0, "sqrt", 0.3, 0.5],
    "model__max_depth": [None, 5, 10, 20, 40],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__min_samples_split": [2, 4, 8, 16],
}

# Tune hyperparameters
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    cv=cv,
    n_iter=50,
    scoring="r2",
    n_jobs=-1,
    random_state=1,
    verbose=1,
)

# Find best model
search.fit(X_train, y_train)
print(f"best model has parameters: {search.best_params_}")
print(f"best model returns score: {search.best_score_}")
best_model = search.best_estimator_

# Get final model evaluation
predictions = best_model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"Final r-squared on test data is: {r2}")

# Feature importance analysis
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

print("\nTop 20 feature importances:")
print(importance_df.head(20))
