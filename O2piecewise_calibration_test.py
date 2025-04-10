import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mass_flow_rate():
    try:
        return float(input("Enter the desired mass flow rate: "))
    except Exception:
        print("Input prompt not supported in this environment. Using a default mass flow rate of 5.0.")
        return 5.0

def get_breakpoints():
    try:
        knee1 = float(input("Enter the first knee point (breakpoint): "))
        knee2 = float(input("Enter the second knee point (breakpoint): "))
        return [knee1, knee2]
    except Exception:
        print("Input prompt not supported in this environment. Using default breakpoints at 33rd and 66th percentiles.")
        return np.percentile(x_data, [33, 66])  # Default to 33rd and 66th percentiles if input fails

# Get the mass flow rate
desired_mass_flow_rate = get_mass_flow_rate()

# Load data from CSV
data = pd.read_csv('O2_800_psi.csv')
x_data = data['Setting'].values
y_data_800 = data['800 psi'].values
y_data_700 = data['700 psi'].values
y_data_600 = data['600 psi'].values
y_data_400 = data['400 psi'].values

# Sort data for plotting and fitting purposes
sorted_indices = np.argsort(x_data)
x_data = x_data[sorted_indices]
y_data_800 = y_data_800[sorted_indices]
y_data_700 = y_data_700[sorted_indices]
y_data_600 = y_data_600[sorted_indices]
y_data_400 = y_data_400[sorted_indices]

# Get the user-defined breakpoints
breakpoints = get_breakpoints()

# Fit linear models to each segment using the user-specified breakpoints
left_mask = x_data <= breakpoints[0]
middle_mask = (x_data > breakpoints[0]) & (x_data <= breakpoints[1])
right_mask = x_data > breakpoints[1]

left_params = np.polyfit(x_data[left_mask], y_data_800[left_mask], 1)
middle_params = np.polyfit(x_data[middle_mask], y_data_800[middle_mask], 1)
right_params = np.polyfit(x_data[right_mask], y_data_800[right_mask], 1)

# Define piecewise function
def piecewise_linear(x):
    if x <= breakpoints[0]:
        return left_params[0] * x + left_params[1]
    elif x <= breakpoints[1]:
        return middle_params[0] * x + middle_params[1]
    else:
        return right_params[0] * x + right_params[1]

# Plot the 3-piece linear segments
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data_800, label='800 psi Data', color='blue')

# Plot each segment separately
plt.plot([min(x_data), breakpoints[0]], 
         [piecewise_linear(min(x_data)), piecewise_linear(breakpoints[0])], 
         color='red', label='3-Piece Linear Approximation')

plt.plot([breakpoints[0], breakpoints[1]], 
         [piecewise_linear(breakpoints[0]), piecewise_linear(breakpoints[1])], 
         color='red')

plt.plot([breakpoints[1], max(x_data)], 
         [piecewise_linear(breakpoints[1]), piecewise_linear(max(x_data))], 
         color='red')

# Shift the lower pressures downward
y_shift_700 = y_data_700[0] - piecewise_linear(x_data[0])
y_shift_600 = y_data_600[0] - piecewise_linear(x_data[0])
y_shift_400 = y_data_400[0] - piecewise_linear(x_data[0])

x_values = np.linspace(min(x_data), max(x_data), 100)

plt.plot(x_values, [piecewise_linear(x) + y_shift_700 for x in x_values], 
         color='green', linestyle='--', label='700 psi Extrapolation')

plt.plot(x_values, [piecewise_linear(x) + y_shift_600 for x in x_values], 
         color='orange', linestyle='--', label='600 psi Extrapolation')

plt.plot(x_values, [piecewise_linear(x) + y_shift_400 for x in x_values], 
         color='purple', linestyle='--', label='400 psi Extrapolation')

plt.scatter(x_data, y_data_700, label='700 psi Data', color='green')
plt.scatter(x_data, y_data_600, label='600 psi Data', color='orange')
plt.scatter(x_data, y_data_400, label='400 psi Data', color='purple')

plt.xlabel('Setting')
plt.ylabel('Mass Flow Rate')
plt.legend()
plt.title('3-Piece Linear Approximation and Extrapolation')
plt.show()