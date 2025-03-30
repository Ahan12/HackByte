import openai
import random
import joblib
import numpy as np
import pandas as pd

# Load the trained model and label encoders
model = joblib.load("waste_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize the OpenAI client with your API key
openai.api_key = ""

# Function to generate meals using the OpenAI API (GPT-4)
def generate_meals_via_api():
    prompt = """
    Generate a list of 20 different meal options for a school cafeteria, considering nutritional balance (vegetables, proteins, grains) and the goal of minimizing waste.
    Each meal should be healthy, easy to prepare, and suitable for a school cafeteria. Format each meal as:
    Meal Name: Brief description of the meal.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    
    # Split response by newline and parse each meal
    meals_data = response.choices[0].message['content'].strip().split("\n")
    meals = []
    for line in meals_data:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        if ": " in line:
            meal_name, meal_desc = line.split(": ", 1)
        else:
            meal_name, meal_desc = line, ""
        meals.append({
            "meal": meal_name.strip(),
            "description": meal_desc.strip(),
            "ingredients": []  # Ingredients list is empty from the API
        })
    return meals

# Default ingredient mapping for known meals (can be refined)
default_ingredients = {
    '17. Shrimp Stir Fry': ['Shrimp', 'Broccoli', 'Soy Sauce', 'Garlic', 'Rice'],
    '5. Beef and Broccoli': ['Beef', 'Broccoli', 'Soy Sauce', 'Rice'],
    '3. Turkey Wrap': ['Turkey', 'Lettuce', 'Tomato', 'Tortilla', 'Cheese'],
    '6. Chicken Caesar Salad': ['Chicken', 'Lettuce', 'Caesar Dressing', 'Parmesan', 'Croutons'],
    '12. Chicken and Vegetable Soup': ['Chicken', 'Carrots', 'Celery', 'Onions', 'Broth']
}

# Helper function: convert a meal dictionary into a feature vector (DataFrame) matching the training schema.
def convert_meal_to_features(meal):
    feature_dict = {
        "Menu_Item_Name": meal["meal"],
        "Food_Category": "Main dish",       # Default category; adjust as needed.
        "Contains_Meat": "No",               # Default value; adjust if parsing ingredients.
        "Contains_Dairy": "No",              # Default value.
        "Contains_Vegetables": "Yes",        # Assume vegetables are present.
        "Is_Fried": "No",                    # Default value.
        "Has_Sauce": "No",                   # Default value.
        "Is_Sweet": "No",                    # Default value.
        "Is_Spicy": "No",                    # Default value.
        "Is_Finger_Food": "No",              # Default value.
        "Is_Preprocessed": "No",             # Default value.
        "Serving_Size_Grams": 200,           # Default serving size.
        "Total_Servings_Prepared": 100       # Default number of servings prepared.
    }
    return pd.DataFrame([feature_dict])

# Function to get waste probability using the trained model.
def get_waste_probability(meal):
    df_sample = convert_meal_to_features(meal)
    
    # Encode categorical variables using the saved label encoders.
    # For any unseen label, replace it with "Unknown" and update the encoder.
    for col, le in label_encoders.items():
        if col in df_sample.columns:
            df_sample[col] = df_sample[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            if "Unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Unknown")
            df_sample[col] = le.transform(df_sample[col])
    
    predicted_waste = model.predict(df_sample)[0]
    return predicted_waste

# Simplified function to check nutritional compliance.
def check_nutritional_compliance(meal):
    # For demonstration purposes, assume all meals are compliant.
    return True

# Function to generate and evaluate meals.
def generate_and_evaluate_menu(num_trials=20):
    candidate_meals = generate_meals_via_api()
    best_meals = []
    
    for _ in range(num_trials):
        # Sample 5 meals to form a weekly plan.
        meals_for_week = random.sample(candidate_meals, 5)
        valid_meals = []
        for meal in meals_for_week:
            if check_nutritional_compliance(meal):
                try:
                    waste_probability = get_waste_probability(meal)
                except Exception as e:
                    print("Error processing meal:", meal, e)
                    continue
                valid_meals.append((meal, waste_probability))
        
        if valid_meals:
            # Select the meal with the lowest predicted waste for this trial.
            valid_meals.sort(key=lambda x: x[1])
            best_meals.append(valid_meals[0])
    
    # Sort all the selected meals and choose the top 5 (lowest waste predictions).
    best_meals.sort(key=lambda x: x[1])
    top_5_meals = best_meals[:5]
    
    # Create final weekly menu.
    weekly_menu = {}
    for meal_tuple in top_5_meals:
        meal_info = meal_tuple[0]
        meal_name = meal_info['meal']
        # If ingredients list is empty, try to fill with default ingredients.
        ingredients = meal_info.get('ingredients', [])
        if not ingredients and meal_name in default_ingredients:
            ingredients = default_ingredients[meal_name]
        weekly_menu[meal_name] = ingredients
        
    return weekly_menu

# Run the process to generate and optimize the weekly meal plan.
if __name__ == "__main__":
    weekly_menu = generate_and_evaluate_menu()
    print("Optimized Weekly Menu:", weekly_menu)
