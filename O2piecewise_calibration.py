import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to get mass flow rate (unchanged)
def get_mass_flow_rate():
    try:
        return float(input("Enter the desired mass flow rate: "))
    except Exception:
        print("Input prompt not supported in this environment. Using a default mass flow rate of 5.0.")
        return 5.0

# Function to get user-selected pressure level
def get_selected_pressure():
    try:
        pressure = int(input("Enter the desired pressure level (800, 700, 600, or 400 psi): "))
        if pressure in [800, 700, 600, 400]:
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
data = pd.read_csv('O2_800_psi.csv', delimiter='\t')  # Fixed delimiter

# Ensure there are at least two columns
if data.shape[1] < 2:
    raise ValueError("Error: O2_800_psi.csv does not have enough columns. Check file formatting.")

x_data = data.iloc[:, 0].values  # Setting
y_data = data.iloc[:, 1].values  # Mass flow rate (800 psi)

# Sort data for plotting and fitting purposes
sorted_indices = np.argsort(x_data)
x_data = x_data[sorted_indices]
y_data = y_data[sorted_indices]

# Get the user-defined breakpoints
breakpoints = [15, 19]

# Fit linear models to each segment ensuring smooth transitions
left_mask = x_data <= breakpoints[0]
middle_mask = (x_data > breakpoints[0]) & (x_data <= breakpoints[1])
right_mask = x_data > breakpoints[1]

if np.sum(left_mask) >= 2:
    left_params = np.polyfit(x_data[left_mask], y_data[left_mask], 1)
else:
    left_params = [0, 0]

if np.sum(middle_mask) >= 2:
    middle_slope, _ = np.polyfit(x_data[middle_mask], y_data[middle_mask], 1)
    middle_intercept = (left_params[0] * breakpoints[0] + left_params[1]) - middle_slope * breakpoints[0]
    middle_params = [middle_slope, middle_intercept]
else:
    middle_params = left_params

if np.sum(right_mask) >= 2:
    right_slope, _ = np.polyfit(x_data[right_mask], y_data[right_mask], 1)
    right_intercept = (middle_params[0] * breakpoints[1] + middle_params[1]) - right_slope * breakpoints[1]
    right_params = [right_slope, right_intercept]
else:
    right_params = middle_params

# Define function for applying the same piecewise linear fit to other pressures
def piecewise_fit(x, offset):
    if x <= breakpoints[0]:
        return left_params[0] * x + left_params[1] + offset
    elif x <= breakpoints[1]:
        return middle_params[0] * x + middle_params[1] + offset
    else:
        return right_params[0] * x + right_params[1] + offset

# Load the second CSV file (new data with multiple pressure levels)
new_data = pd.read_csv('O2_all_press.csv', delimiter=',')

# Ensure there are at least five columns
if new_data.shape[1] < 5:
    raise ValueError("Error: O2_all_press.csv does not have enough columns. Check file formatting.")

new_settings = new_data.iloc[:, 0].values  # Setting column
pressure_levels = {
    800: new_data.iloc[:, 1].values,
    700: new_data.iloc[:, 2].values,
    600: new_data.iloc[:, 3].values,
    400: new_data.iloc[:, 4].values
}

# Generate smooth x-values for curve plotting
x_smooth = np.linspace(min(x_data), max(x_data), 500)

# Generate piecewise linear predictions for each pressure level
predicted_curves = {}
for pressure, values in pressure_levels.items():
    offset = np.mean(values - pressure_levels[800])
    predicted_curves[pressure] = np.array([piecewise_fit(x, offset) for x in x_smooth])

# Determine the lowest setting for the user-selected pressure
def find_optimal_setting():
    for x in x_smooth:
        if piecewise_fit(x, np.mean(pressure_levels[selected_pressure] - pressure_levels[800])) >= desired_mass_flow_rate:
            return x
    return None

optimal_setting = find_optimal_setting()
print(f"Optimal setting for {desired_mass_flow_rate} at {selected_pressure} psi: {optimal_setting}")

# Plot the original data, piecewise linear approximation, and new data points
plt.figure(figsize=(10, 8))

plt.scatter(x_data, y_data, label='Original Data (800 psi)', color='blue')
plt.plot(x_smooth, predicted_curves[800], color='red', label='Piecewise Fit (800 psi)')
plt.plot(x_smooth, predicted_curves[700], color='purple', linestyle='dashed', label='Piecewise Fit (700 psi)')
plt.plot(x_smooth, predicted_curves[600], color='orange', linestyle='dashed', label='Piecewise Fit (600 psi)')
plt.plot(x_smooth, predicted_curves[400], color='brown', linestyle='dashed', label='Piecewise Fit (400 psi)')

# Highlight the optimal setting on the plot
if optimal_setting is not None:
    plt.scatter(optimal_setting, desired_mass_flow_rate, color='black', s=200, marker='*',
                label=f'Optimal Setting: {optimal_setting:.2f} (Pressure: {selected_pressure} psi)')

plt.xlabel('Setting')
plt.ylabel('Mass Flow Rate')
plt.legend()
plt.title('Piecewise Linear Approximation with Extrapolated New Data')
plt.grid(True)
plt.show()
