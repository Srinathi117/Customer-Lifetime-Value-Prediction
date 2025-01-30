import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the CSV file
df = pd.read_csv('C:/Users/aravi/OneDrive/Desktop/codeClause/project 2/customer_data.csv')  # Make sure the file path is correct

# Preview the dataset to ensure it's loaded correctly
print(df.head())

# Drop Customer ID as it's not relevant for the prediction
df.drop(columns=['Customer ID'], inplace=True)

# Define the features (X) and target (y)
X = df.drop(columns=['CLV (Target)'])
y = df['CLV (Target)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.title('Actual vs Predicted CLV')
plt.show()
