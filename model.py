# transfer_learning.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def perform_transfer_learning(file_path):
    # Load the new dataset with the appended data
    df = pd.read_csv(file_path)

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
        df[col] = df[col].astype(str)
        le = label_encoders.get(col)
        
        if le is None:
            le = LabelEncoder()
            le.fit(df[col])
            label_encoders[col] = le
        else:
            new_classes = df[col].unique()
            le.classes_ = np.unique(np.concatenate([le.classes_.astype(str), new_classes.astype(str)]))
        
        df[col] = le.transform(df[col])
        df[col] = df[col].astype(float)

    # Save the updated label encoders
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
    print(f"Test Score: {score}")

    # Save the retrained model and updated label encoders
    joblib.dump(pipeline, "retrained_waste_prediction_model.pkl")  # Save the retrained model
    joblib.dump(label_encoders, "updated_label_encoders.pkl")  # Save the updated label encoders

    print("Retrained model and updated encoders saved successfully!")
