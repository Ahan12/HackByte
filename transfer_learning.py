import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the new dataset with the appended data
file_path = "correlated_school_cafeteria_food_waste.csv"  # Update with the correct file path
df = pd.read_csv(file_path)

# Print columns to ensure the structure is as expected
print(df.columns)

# Drop the 'Date' column as it's not used for prediction
df = df.drop(columns=["Date"])

# List of categorical columns used in training
categorical_columns = [
    "Menu_Item_Name", "Food_Category", "Contains_Meat", "Contains_Dairy",
    "Contains_Vegetables", "Is_Fried", "Has_Sauce", "Is_Sweet", "Is_Spicy",
    "Is_Finger_Food", "Is_Preprocessed"
]

# Load the previously saved label encoders
label_encoders = joblib.load("label_encoders.pkl")

# Update the label encoders to include new labels (if any)
for col in categorical_columns:
    # Ensure that all values in the column are strings to avoid mixed types
    df[col] = df[col].astype(str)
    
    # Get the current encoder for the column
    le = label_encoders.get(col)
    
    if le is None:
        # If the encoder doesn't exist, create and fit it on the current data
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le  # Save the newly trained encoder
    else:
        # Update the encoder to include any new labels that might be in the new data
        # Note: We convert both old and new classes to strings, just in case.
        new_classes = df[col].unique()
        le.classes_ = np.unique(np.concatenate([le.classes_.astype(str), new_classes.astype(str)]))
    
    # Transform the column using the updated encoder
    df[col] = le.transform(df[col])
    
    # Explicitly cast the column to float to ensure StandardScaler gets numeric data
    df[col] = df[col].astype(float)

# Save the updated label encoders to ensure they reflect the new data
joblib.dump(label_encoders, "updated_label_encoders.pkl")

# Define features (X) and target (y)
X = df.drop(columns=["Total_Servings_Wasted"])
y = df["Total_Servings_Wasted"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the previously saved pipeline (this includes scaling and the RandomForest model)
pipeline = joblib.load("waste_prediction_model.pkl")

# Re-train the pipeline with the new data
pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)

# Save the retrained pipeline and updated label encoders
joblib.dump(pipeline, "retrained_waste_prediction_model.pkl")  # Save the retrained model
joblib.dump(label_encoders, "updated_label_encoders.pkl")  # Save the updated label encoders

print("Retrained model and updated encoders saved successfully!")
