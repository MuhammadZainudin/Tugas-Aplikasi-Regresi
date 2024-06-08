import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('Student_Performance.csv')

X = df[['Hours Studied']].values
y = df['Performance Index'].values

linear_model = LinearRegression()
linear_model.fit(X, y)

y_pred_linear = linear_model.predict(X)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear Regression')
plt.legend()
plt.show()

print(f"Koefisien linear: {linear_model.coef_[0]}")
print(f"Intercept linear: {linear_model.intercept_}")

mse_linear = mean_squared_error(y, y_pred_linear)
rms_linear = np.sqrt(mse_linear)
print(f"MSE Linear Regression: {mse_linear}")
print(f"RMS Linear Regression: {rms_linear}")

# Testing
print()
print("Tesing")
test_hours = [[2], [4], [6]]  
predicted_scores_linear = linear_model.predict(test_hours)
print("Predicted Scores (Linear Regression):")
for i, hours in enumerate(test_hours):
    print(f"Hours Studied: {hours[0]}, Predicted Score: {predicted_scores_linear[i]}")
