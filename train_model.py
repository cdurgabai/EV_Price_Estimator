import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Load dataset
df = pd.read_csv(os.path.join('data', 'train.csv'))

# Drop unnecessary columns
df.drop(columns=[
    'ID', 'VIN (1-10)', 'DOL Vehicle ID',
    'Vehicle Location', 'County', 'State', 'ZIP Code'
], inplace=True, errors='ignore')

# Clean target column
df['Expected Price ($1k)'] = pd.to_numeric(df['Expected Price ($1k)'], errors='coerce')
df = df[df['Expected Price ($1k)'].notna()]

# Features and target
y = df['Expected Price ($1k)']
X = df.drop(columns=['Expected Price ($1k)'])

# Feature types
numerical_features = ['Model Year', 'Electric Range', 'Base MSRP']
categorical_features = [
    'Make', 'Model', 'Electric Vehicle Type',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility',
    'City', 'Electric Utility'
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models to train (No Linear Regression)
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

model_scores = {}

# Train and evaluate
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluation
    model_scores[model_name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
    }

    # Save model
    joblib.dump(pipeline, f'{model_name.lower()}_ev_price_model.pkl')

# Print results
print("Model Performance Metrics:")
for model_name, scores in model_scores.items():
    print(f"\n{model_name} Performance:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

# Visual comparison
models_list = list(model_scores.keys())
mae_list = [model_scores[m]["MAE"] for m in models_list]
mse_list = [model_scores[m]["MSE"] for m in models_list]
r2_list = [model_scores[m]["R²"] for m in models_list]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].bar(models_list, mae_list, color='skyblue')
axs[0].set_title('Mean Absolute Error (MAE)')
axs[1].bar(models_list, mse_list, color='lightgreen')
axs[1].set_title('Mean Squared Error (MSE)')
axs[2].bar(models_list, r2_list, color='salmon')
axs[2].set_title('R² Score')
plt.tight_layout()
plt.show()

# Save best model
best_model_name = max(model_scores, key=lambda x: model_scores[x]['R²'])
best_model_pipeline = joblib.load(f'{best_model_name.lower()}_ev_price_model.pkl')
print(f"\nBest model based on R² score: {best_model_name}")
