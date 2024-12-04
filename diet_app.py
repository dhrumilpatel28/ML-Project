import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the trained model

# Load pre-trained BMI model and food dataset
bmi_model_path = 'bmi_model.pkl'  # Replace with your trained model path
food_data_path = './cleaned_dataset.csv'  # Replace with your cleaned dataset path

# Load the BMI model and food dataset
bmi_model = joblib.load(bmi_model_path)
food_df = pd.read_csv(food_data_path)

# Clean column names for consistency
food_df.columns = food_df.columns.str.strip()

# BMI Class Mapping
bmi_label = {
    0: "Underweight",
    1: "Normal weight",
    2: "Overweight",
    3: "Obese class 1",
    4: "Obese class 2",
    5: "Obese class 3"
}


# Function to predict BMI class
def predict_bmi(age, weight, height):
    user_input = np.array([[age, weight, height]])
    bmi_class = bmi_model.predict(user_input)
    return bmi_class[0]


# Function to recommend diet
def recommend_diet(bmi_class, food_df):
    if bmi_class == 0:  # Underweight
        filtered_food = food_df[food_df['Caloric Value'] > 300]  # High-calorie foods for weight gain
    elif bmi_class == 1:  # Normal weight
        user_choice = st.radio("Do you want to gain weight?", ("No", "Yes"))
        if user_choice == "Yes":
            filtered_food = food_df[food_df['Caloric Value'] > 300]
        else:
            filtered_food = food_df  # Balanced diet
    elif bmi_class in [2, 3, 4, 5]:  # Overweight or Obese classes
        filtered_food = food_df[food_df['Caloric Value'] < 200]  # Low-calorie foods for weight loss
    else:
        st.error("Unexpected BMI class. Unable to provide recommendations.")
        return None, None, None

    # Ensure total daily calories do not exceed 3000
    breakfast = filtered_food.sample(1)
    lunch = filtered_food.sample(1)
    dinner = filtered_food.sample(1)
    total_calories = (
        breakfast['Caloric Value'].values[0]
        + lunch['Caloric Value'].values[0]
        + dinner['Caloric Value'].values[0]
    )

    while total_calories > 3000:
        # Resample meals if calories exceed the limit
        breakfast = filtered_food.sample(1)
        lunch = filtered_food.sample(1)
        dinner = filtered_food.sample(1)
        total_calories = (
            breakfast['Caloric Value'].values[0]
            + lunch['Caloric Value'].values[0]
            + dinner['Caloric Value'].values[0]
        )

    return breakfast, lunch, dinner


# Streamlit App
st.title("Diet Plan Recommendation System")

# User inputs
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0, step=0.1)
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)

# Predict BMI class and generate diet plan
if st.sidebar.button("Get Recommendations"):
    bmi_class = predict_bmi(age, weight, height)
    st.write(f"Your BMI Class: {bmi_label[bmi_class]}")

    # Recommend diet based on BMI class
    breakfast, lunch, dinner = recommend_diet(bmi_class, food_df)

    if breakfast is not None:
        st.write("### Recommended Diet Plan:")
        st.write("**Breakfast**")
        st.table(breakfast[['food', 'Caloric Value', 'Fat']])
        st.write("**Lunch**")
        st.table(lunch[['food', 'Caloric Value', 'Fat']])
        st.write("**Dinner**")
        st.table(dinner[['food', 'Caloric Value', 'Fat']])
