# conversion.py
import pandas as pd
from datetime import datetime
import openai
import re

# Set your OpenAI API key (replace with your actual key)
openai.api_key = ""
def get_item_attributes(item_name):
    # For apple, return hardcoded attributes ensuring proper values.
    if item_name.strip().lower() == "apple":
        return {
            "Food Category": "Fruit",
            "Contains Meat": "No",
            "Contains Dairy": "No",
            "Contains Vegetables": "No",
            "Is Fried": "No",
            "Has Sauce": "No",
            "Is Sweet": "Yes",
            "Is Spicy": "No",
            "Is Finger Food": "Yes",
            "Is Preprocessed": "No",
            "Serving Size in grams": "182"
        }
    # Otherwise, query OpenAI's API
    prompt = (
        f"Provide detailed attributes for the following food item detected in a cafeteria: {item_name}. "
        "Return the answer in the following format exactly, with one field per line:\n"
        "Food Category: <value>\n"
        "Contains Meat: <Yes/No>\n"
        "Contains Dairy: <Yes/No>\n"
        "Contains Vegetables: <Yes/No>\n"
        "Is Fried: <Yes/No>\n"
        "Has Sauce: <Yes/No>\n"
        "Is Sweet: <Yes/No>\n"
        "Is Spicy: <Yes/No>\n"
        "Is Finger Food: <Yes/No>\n"
        "Is Preprocessed: <Yes/No>\n"
        "Serving Size in grams: <number>"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        attributes_text = response['choices'][0]['message']['content'].strip()
        attribute_dict = {}
        for line in attributes_text.split('\n'):
            if ':' in line:
                key, value = line.split(":", 1)
                attribute_dict[key.strip()] = value.strip()
        return attribute_dict
    except Exception as e:
        print(f"Error retrieving attributes for {item_name}: {e}")
        return None

def process_with_time_window(detection_log):
    detection_log['Timestamp'] = pd.to_datetime(detection_log['Timestamp'])
    # Sort by the Detection column (we assume it includes the object name) and then Timestamp.
    detection_log = detection_log.sort_values(by=['Detection', 'Timestamp'])
    seen = {}
    filtered_log = []
    for _, row in detection_log.iterrows():
        # Extract the object name from the Detection column (e.g. "apple (0.58)" -> "apple")
        detection_name = row['Detection'].split('(')[0].strip()
        timestamp = row['Timestamp']
        if detection_name not in seen:
            seen[detection_name] = timestamp
            filtered_log.append(row)
        else:
            time_diff = (timestamp - seen[detection_name]).total_seconds()
            if time_diff > 5:
                seen[detection_name] = timestamp
                filtered_log.append(row)
    return pd.DataFrame(filtered_log)

def process_detections(detection_log_path, output_file="correlated_output.csv"):
    # Load the detection log (new format)
    detection_log = pd.read_csv(detection_log_path)
    # Apply 5-second window filtering
    filtered_detection_log = process_with_time_window(detection_log)
    # Get unique object names from the Detection column, stripping extra spaces.
    unique_classes = filtered_detection_log['Detection'].apply(lambda x: x.split('(')[0].strip()).unique()
    item_details = {}
    for item_class in unique_classes:
        attributes = get_item_attributes(item_class)
        item_details[item_class] = attributes

    output_data = []
    for cls in unique_classes:
        attributes = item_details.get(cls)
        if attributes:
            # Count how many times this object appears (after filtering)
            servings_wasted = filtered_detection_log[
                filtered_detection_log['Detection'].apply(lambda x: x.split('(')[0].strip() == cls)
            ].shape[0]
            servings_served = 0  # as per your assumption
            output_data.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Menu_Item_Name': cls,
                'Food_Category': attributes.get('Food Category', 'Unknown'),
                'Contains Meat': attributes.get('Contains Meat', 'No'),
                'Contains Dairy': attributes.get('Contains Dairy', 'No'),
                'Contains Vegetables': attributes.get('Contains Vegetables', 'No'),
                'Is Fried': attributes.get('Is Fried', 'No'),
                'Has Sauce': attributes.get('Has Sauce', 'No'),
                'Is Sweet': attributes.get('Is Sweet', 'No'),
                'Is Spicy': attributes.get('Is Spicy', 'No'),
                'Is Finger Food': attributes.get('Is Finger Food', 'No'),
                'Is Preprocessed': attributes.get('Is Preprocessed', 'No'),
                'Serving_Size_Grams': int(attributes.get('Serving Size in grams', '0')),
                'Total_Servings_Wasted': servings_wasted,
                'Total_Servings_Served': servings_served
            })
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Conversion complete. Saved as {output_file}.")
    
    # Now append the new data to the dataset.csv
    append_to_dataset(output_file, 'correlated_school_cafeteria_food_waste.csv')

def append_to_dataset(new_data_file, dataset_file):
    # Read both the new data and the existing dataset
    new_data = pd.read_csv(new_data_file)
    try:
        dataset = pd.read_csv(dataset_file)
        # Append the new data to the existing dataset
        combined_data = pd.concat([dataset, new_data], ignore_index=True)
    except FileNotFoundError:
        # If dataset.csv doesn't exist, create a new one with the new data
        combined_data = new_data

    # Save the combined dataset back to dataset.csv
    combined_data.to_csv(dataset_file, index=False)
    print(f"Appended data to {dataset_file}")
