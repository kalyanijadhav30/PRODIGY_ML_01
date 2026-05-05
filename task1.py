import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_csv("train.csv")

# Step 2: Select important columns
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

# Step 3: Handle missing values
X = X.fillna(0)

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Create model
model = LinearRegression()

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predict
predictions = model.predict(X_test)

# Step 8: Evaluate
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Step 9: Show predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {int(predictions[i])}")

# Step 10: Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
