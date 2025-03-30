import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime
fake = Faker()
# Expanded lists of food items
main_dishes = [
    "Spaghetti Bolognese", "Fried Chicken", "Cheeseburger", "Grilled Cheese Sandwich", "Chicken Nuggets",
    "Beef Tacos", "BBQ Pulled Pork Sandwich", "Veggie Burger", "Chicken Alfredo", "Teriyaki Chicken Bowl",
    "Turkey Sandwich", "Meatball Sub", "Fish Fillet Sandwich", "Chili Con Carne", "Stuffed Bell Peppers"
]

sides = [
    "French Fries", "Mashed Potatoes", "Rice Pilaf", "Mac and Cheese", "Garlic Bread",
    "Cornbread", "Coleslaw", "Potato Wedges", "Baked Beans", "Sweet Potato Fries"
]

vegetables = [
    "Steamed Broccoli", "Carrot Sticks", "Green Beans", "Mixed Salad", "Roasted Veggies",
    "Zucchini Sauté", "Peas and Carrots", "Cauliflower Bites", "Spinach Salad", "Cucumber Slices"
]

fruits = [
    "Apple Slices", "Banana", "Fruit Cup", "Orange Wedges", "Grapes",
    "Pear Slices", "Pineapple Chunks", "Watermelon Cubes", "Strawberry Halves", "Mixed Berries"
]

desserts = [
    "Chocolate Pudding", "Apple Pie", "Ice Cream Cup", "Cookies", "Brownie",
    "Vanilla Yogurt", "Rice Krispie Treat", "Cupcake", "Banana Bread", "Fruit Tart"
]
# Generate ~200 school days (weekdays from Sep to Jun)
def generate_school_days(start_date, end_date):
    days = pd.date_range(start=start_date, end=end_date, freq='B')
    return days[:200]

school_days = generate_school_days("2023-09-01", "2024-06-30")
# Generate dataset
records = []

for day in school_days:
    categories_and_counts = [
        ("Main dish", 2, main_dishes),
        ("Side", 2, sides),
        ("Vegetable", 1, vegetables),
        ("Fruit", 1, fruits),
        ("Dessert", 2, desserts)
    ]

    for category, count, item_list in categories_and_counts:
        for _ in range(count):
            item = random.choice(item_list)
            if category == "Main dish":
                size = random.randint(200, 300)
                prepared = random.randint(150, 300)
            elif category == "Side":
                size = random.randint(100, 150)
                prepared = random.randint(200, 350)
            elif category == "Vegetable":
                size = random.randint(80, 120)
                prepared = random.randint(100, 250)
            elif category == "Fruit":
                size = random.randint(80, 150)
                prepared = random.randint(100, 250)
            else:
                size = random.randint(100, 180)
                prepared = random.randint(200, 350)

            contains_meat = "Yes" if any(meat in item.lower() for meat in ["chicken", "beef", "burger", "pork", "meatball", "fillet", "turkey", "taco", "bolognese", "carne"]) else "No"
            contains_dairy = "Yes" if any(dairy in item.lower() for dairy in ["cheese", "cream", "milk", "pudding", "ice cream", "mac", "yogurt"]) else "No"
            contains_veg = "Yes" if category in ["Vegetable", "Side"] or any(veg in item.lower() for veg in ["salad", "broccoli", "carrot", "beans", "veggies", "zucchini", "peas", "spinach", "cucumber"]) else "No"
            is_fried = "Yes" if any(fried in item.lower() for fried in ["fried", "nuggets", "fries", "fillet", "wedges"]) else "No"
            has_sauce = "Yes" if any(sauce in item.lower() for sauce in ["spaghetti", "bolognese", "alfredo", "gravy", "teriyaki", "chili"]) else "No"
            is_sweet = "Yes" if category == "Dessert" or any(sweet in item.lower() for sweet in ["fruit", "pudding", "cookies", "pie", "ice cream", "cupcake", "banana bread", "tart"]) else "No"
            is_spicy = "Yes" if any(spice in item.lower() for spice in ["chili", "spicy", "taco"]) else ("Yes" if random.random() < 0.03 else "No")
            is_finger_food = "Yes" if any(food in item.lower() for food in ["nuggets", "sticks", "wedges", "cookies", "fries", "grapes", "slices", "cup", "fillet", "bread"]) else "No"
            is_preprocessed = "Yes" if random.random() < 0.4 else "No"

            waste_rate = 0.05 + (0.1 if is_spicy == "Yes" else 0) + (0.1 if is_preprocessed == "Yes" else 0)
            waste_rate -= 0.05 if is_finger_food == "Yes" else 0
            waste_rate = np.clip(waste_rate + np.random.normal(0, 0.05), 0, 0.5)
            wasted = int(prepared * waste_rate)

            records.append({
                "Date": day.strftime("%Y-%m-%d"),
                "Menu_Item_Name": item,
                "Food_Category": category,
                "Contains_Meat": contains_meat,
                "Contains_Dairy": contains_dairy,
                "Contains_Vegetables": contains_veg,
                "Is_Fried": is_fried,
                "Has_Sauce": has_sauce,
                "Is_Sweet": is_sweet,
                "Is_Spicy": is_spicy,
                "Is_Finger_Food": is_finger_food,
                "Is_Preprocessed": is_preprocessed,
                "Serving_Size_Grams": size,
                "Total_Servings_Prepared": prepared,
                "Total_Servings_Wasted": wasted
            })

df = pd.DataFrame(records)
df.head()
df.to_csv("expanded_school_cafeteria_food_waste.csv", index=False)
print("CSV file saved as expanded_school_cafeteria_food_waste.csv")