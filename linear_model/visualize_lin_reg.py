import numpy as np
import matplotlib.pyplot as plt
from .linear_regression import LinearRegression

# noisy data that looks like a line
X = 2 * np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
y = y.squeeze()

model = LinearRegression()
model.fit(X, y)

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')

plt.plot(X_line, y_line, color='red', linewidth=2, label='Model Prediction')

plt.xlabel('Input Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()
