import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('Student_Performance.csv')

X = df[['Hours Studied']].values
y = df['Performance Index'].values

X_squared = X ** 2  
quad_model = LinearRegression()
quad_model.fit(X_squared, y)

y_pred_quad = quad_model.predict(X_squared)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(np.sort(X, axis=0), y_pred_quad[np.argsort(X, axis=0)][:,0], color='green', label='Quadratic Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Quadratic Regression')
plt.legend()
plt.show()

print(f"Koefisien quadratic: {quad_model.coef_[0]}")
print(f"Intercept quadratic: {quad_model.intercept_}")

mse_quad = mean_squared_error(y, y_pred_quad)
rms_quad = np.sqrt(mse_quad)
print(f"MSE Quadratic Regression: {mse_quad}")
print(f"RMS Quadratic Regression: {rms_quad}")

# Testing
print()
print("Tesing")
test_hours = [[2], [4], [6]]  
test_hours_squared = np.array(test_hours) ** 2
predicted_scores_quad = quad_model.predict(test_hours_squared)
print("Predicted Scores (Quadratic Regression):")
for i, hours in enumerate(test_hours):
    print(f"Hours Studied: {hours[0]}, Predicted Score: {predicted_scores_quad[i]}")
