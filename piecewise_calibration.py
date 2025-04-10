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
x_data = data.iloc[:, 0].values  # Setting
y_data = data.iloc[:, 1].values  # Mass flow rate

# Sort data for plotting and fitting purposes
sorted_indices = np.argsort(x_data)
x_data = x_data[sorted_indices]
y_data = y_data[sorted_indices]

# Get the user-defined breakpoints
breakpoints = get_breakpoints()

# Fit linear models to each segment using the user-specified breakpoints
left_mask = x_data <= breakpoints[0]
middle_mask = (x_data > breakpoints[0]) & (x_data <= breakpoints[1])
right_mask = x_data > breakpoints[1]

left_params = np.polyfit(x_data[left_mask], y_data[left_mask], 1)
middle_params = np.polyfit(x_data[middle_mask], y_data[middle_mask], 1)
right_params = np.polyfit(x_data[right_mask], y_data[right_mask], 1)

# Define piecewise function
def piecewise_linear(x):
    if x <= breakpoints[0]:
        return left_params[0] * x + left_params[1]
    elif x <= breakpoints[1]:
        return middle_params[0] * x + middle_params[1]
    else:
        return right_params[0] * x + right_params[1]

# Calculate setting for the desired mass flow rate using the piecewise function
corresponding_setting = None
if desired_mass_flow_rate <= piecewise_linear(breakpoints[0]):
    corresponding_setting = (desired_mass_flow_rate - left_params[1]) / left_params[0]
elif desired_mass_flow_rate <= piecewise_linear(breakpoints[1]):
    corresponding_setting = (desired_mass_flow_rate - middle_params[1]) / middle_params[0]
else:
    corresponding_setting = (desired_mass_flow_rate - right_params[1]) / right_params[0]

# Plot the 3-piece linear segments
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data', color='blue')

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

# Mark the calculated setting for the desired mass flow rate
plt.scatter(corresponding_setting, desired_mass_flow_rate, color='green', s=100, 
            label=f'Setting for {desired_mass_flow_rate:.2f}: {corresponding_setting:.2f}')
plt.xlabel('Setting')
plt.ylabel('Mass Flow Rate')
plt.legend()
plt.title('3-Piece Linear Approximation for Setting vs. Mass Flow Rate')
plt.show()
