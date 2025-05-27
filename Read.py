import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('ML Project/synthetic_macro_dataset.csv')

# Ensure your dataset contains the expected columns
expected_columns = ['age', 'gender', 'height', 'weight', 'activity_level', 'goal']
if not all(col in data.columns for col in expected_columns):
    raise ValueError("Dataset missing required columns.")

# Define the features (X) and target (y)
X = data[expected_columns]
y = data[['protein', 'carbs', 'fats', 'water']]

# Define categorical and numeric columns
categorical_columns = ['gender', 'activity_level', 'goal']
numeric_columns = ['age', 'height', 'weight']

# Set up preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ]
)

# Build the full pipeline with a RandomForestRegressor model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Save the model to disk
with open('macro_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print("Model saved successfully!")

# Optionally, print a score to see how well the model performs on the test set
print(f"Model R² score: {model.score(X_test, y_test)}")
