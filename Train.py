
import pandas as pd
import numpy as np
import random

activity_levels = ['sedentary', 'moderate', 'active']
goals = ['cutting', 'maintenance', 'bulking']
genders = ['male', 'female']

def calculate_macros(row):
    weight = row['weight']
    activity = row['activity_level']
    goal = row['goal']
    gender = row['gender']

    if goal == 'cutting':
        protein = weight * 2.2
        carbs = weight * 2.5
        fats = weight * 0.8
    elif goal == 'maintenance':
        protein = weight * 2.0
        carbs = weight * 3.5
        fats = weight * 1.0
    else:  
        protein = weight * 2.0
        carbs = weight * 5.0
        fats = weight * 1.2

    if activity == 'moderate':
        carbs *= 1.1
        fats *= 1.1
    elif activity == 'active':
        carbs *= 1.25
        fats *= 1.25

    water = weight * 0.035
    if activity == 'moderate':
        water += 0.3
    elif activity == 'active':
        water += 0.6

    return pd.Series([round(protein,1), round(carbs,1), round(fats,1), round(water,2)])

samples = 10000
data = {
    'age': np.random.randint(18, 60, samples),
    'gender': [random.choice(genders) for _ in range(samples)],
    'height': np.random.randint(150, 200, samples),
    'weight': np.random.randint(45, 100, samples),
    'activity_level': [random.choice(activity_levels) for _ in range(samples)],
    'goal': [random.choice(goals) for _ in range(samples)],
}

df = pd.DataFrame(data)
df[['protein', 'carbs', 'fats', 'water']] = df.apply(calculate_macros, axis=1)

df.to_csv("synthetic_macro_dataset.csv", index=False)
print("✅ Dataset generated and saved as 'synthetic_macro_dataset.csv'")
