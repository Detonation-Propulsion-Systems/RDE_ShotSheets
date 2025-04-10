import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to get mass flow rate
def get_mass_flow_rate():
    try:
        return float(input("Enter the desired mass flow rate: "))
    except Exception:
        print("Input prompt not supported in this environment. Using a default mass flow rate of 5.0.")
        return 5.0

# Function to get user-selected pressure level
def get_selected_pressure():
    try:
        pressure = int(input("Enter the desired pressure level (800, 700, 600, 500, or 400 psi): "))
        if pressure in [800, 700, 600, 500, 400]:
            return pressure
        else:
            print("Invalid input. Defaulting to 800 psi.")
            return 800
    except Exception:
        print("Input prompt not supported in this environment. Using a default pressure level of 800 psi.")
        return 800

# Get user inputs
desired_mass_flow_rate = get_mass_flow_rate()
selected_pressure = get_selected_pressure()

# Load the first CSV file (original data)
data = pd.read_csv('C2H4_800_psi.csv', delimiter=',')

# Debugging: Print the first few rows to check the structure
print("C2H4_800_psi.csv content preview:")
print(data.head())
print("Column names:", data.columns)

# Check if there are at least two columns
if data.shape[1] < 2:
    raise ValueError("Error: C2H4_800_psi.csv does not have enough columns. Check file formatting.")

# Ensure there are at least two columns
if data.shape[1] < 2:
    raise ValueError("Error: C2H4_800_psi.csv does not have enough columns. Check file formatting.")

x_data = data.iloc[:, 0].values  # Setting
y_data = data.iloc[:, 1].values  # Mass flow rate (800 psi)

# Sort data for plotting and fitting purposes
sorted_indices = np.argsort(x_data)
x_data = x_data[sorted_indices]
y_data = y_data[sorted_indices]

# Fit a single linear regression model to the dataset
slope, intercept = np.polyfit(x_data, y_data, 1)

def linear_fit(x, offset):
    return slope * x + intercept + offset

# Load the second CSV file (new data with multiple pressure levels)
new_data = pd.read_csv('C2H4_all_press.csv', delimiter=',')

# Ensure there are at least six columns (including 500 psi)
if new_data.shape[1] < 6:
    raise ValueError("Error: C2H4_all_press.csv does not have enough columns. Check file formatting.")

new_settings = new_data.iloc[:, 0].values  # Setting column
pressure_levels = {
    800: new_data.iloc[:, 1].values,
    700: new_data.iloc[:, 2].values,
    600: new_data.iloc[:, 3].values,
    500: new_data.iloc[:, 4].values,
    400: new_data.iloc[:, 5].values
}

# Sort data for plotting and fitting purposes
sorted_indices = np.argsort(new_settings)
new_settings = new_settings[sorted_indices]
for key in pressure_levels:
    pressure_levels[key] = pressure_levels[key][sorted_indices]

# Generate smooth x-values for plotting (allowing for negative values)
x_smooth = np.linspace(-10, max(x_data), 500)

# Compute linear fit predictions for all pressures
predicted_curves = {}
for pressure in pressure_levels:
    offset = np.mean(pressure_levels[pressure] - pressure_levels[800])
    predicted_curves[pressure] = np.array([linear_fit(x, offset) for x in x_smooth])

# Determine the lowest setting for the user-selected pressure
def find_optimal_setting():
    for x in x_smooth:
        if linear_fit(x, np.mean(pressure_levels[selected_pressure] - pressure_levels[800])) >= desired_mass_flow_rate:
            return x
    return None

optimal_setting = find_optimal_setting()
print(f"Optimal setting for {desired_mass_flow_rate} at {selected_pressure} psi: {optimal_setting}")

# Plot the original data, linear approximation, and new data points
plt.figure(figsize=(10, 8))

plt.scatter(x_data, y_data, label='Original Data (800 psi)', color='blue')
plt.plot(x_smooth, predicted_curves[800], color='red', label='Linear Fit (800 psi)')
plt.plot(x_smooth, predicted_curves[700], color='purple', linestyle='dashed', label='Linear Fit (700 psi)')
plt.plot(x_smooth, predicted_curves[600], color='orange', linestyle='dashed', label='Linear Fit (600 psi)')
plt.plot(x_smooth, predicted_curves[500], color='green', linestyle='dashed', label='Linear Fit (500 psi)')
plt.plot(x_smooth, predicted_curves[400], color='brown', linestyle='dashed', label='Linear Fit (400 psi)')

# Highlight the optimal setting on the plot
if optimal_setting is not None:
    plt.scatter(optimal_setting, desired_mass_flow_rate, color='black', s=200, marker='*',
                label=f'Optimal Setting: {optimal_setting:.2f} (Pressure: {selected_pressure} psi)')

plt.xlabel('Setting')
plt.ylabel('Mass Flow Rate')
plt.legend()
plt.title('Linear Calibration for C2H4 with Extrapolated Pressure Data')
plt.grid(True)
plt.show()
