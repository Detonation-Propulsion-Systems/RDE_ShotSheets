import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def get_mass_flow_rate():
    try:
        return float(input("Enter the desired mass flow rate: "))
    except Exception:
        print("Input prompt not supported in this environment. Using a default mass flow rate of 5.0.")
        return 5.0  # Set a default if input isn't supported

# Get the mass flow rate at the beginning
desired_mass_flow_rate = get_mass_flow_rate()

# Load data from CSV
data = pd.read_csv('O2_800_psi.csv')  # Using the correct file name
x_data = data.iloc[:, 0].values  # Setting (x-axis)
y_data = data.iloc[:, 1].values  # Mass flow rate (y-axis)

# Define a sigmoid function
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

# Initial guess based on data
initial_guess = [max(y_data), np.median(x_data), 1, min(y_data)]

# Fit the sigmoid curve with increased maxfev
try:
    params, covariance = curve_fit(sigmoid, x_data, y_data, p0=initial_guess, maxfev=5000)
except RuntimeError as e:
    print(f"Fit failed: {e}")
    params = initial_guess  # Fallback to initial guess

# Generate x values for the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit = sigmoid(x_fit, *params)

# Find the closest point on the sigmoid curve to the desired mass flow rate
closest_idx = np.argmin(np.abs(y_fit - desired_mass_flow_rate))
corresponding_setting = x_fit[closest_idx]

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data', color='blue')
plt.plot(x_fit, y_fit, label='Fitted Sigmoid', color='red')
plt.scatter(corresponding_setting, desired_mass_flow_rate, color='green', s=100, label=f'Setting for {desired_mass_flow_rate:.2f}: {corresponding_setting:.2f}')
plt.xlabel('Setting')
plt.ylabel('Mass Flow Rate')
plt.legend()
plt.title('Sigmoid Fit for Setting vs. Mass Flow Rate')
plt.show()
