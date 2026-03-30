# Program to find out the Student Performance Predictor

#import the various required libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load the dataset
df = pd.read_csv('data/student_data.csv')

# Remove the spaces
df.columns = df.columns.str.strip()

# Print columns
print("Columns in dataset :", df.columns)

# Features (inputs) and target (output)
X = df[['HoursStudy','attendance']]
y = df['marks']

plt.scatter(df['HoursStudy'],df['marks'])
plt.xlabel('StudyHours')
plt.ylabel('marks')
plt.show()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,random_state = 42)


# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model R^2 Score :",model.score(X_test , y_test))

# Ask for user input
HoursStudy = int(input("Enter the number of study hours : "))
attendance = float(input("Enter the student's attendance in percentage :"))

# Predict result
prediction = model.predict([[HoursStudy, attendance]])

# Print output
print(f"Predicted Marks: {prediction[0]:.2f}")