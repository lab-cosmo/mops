import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
with open("strong_openmp.out", 'r') as file:
    lines = file.readlines()

# Filter out lines that contain 'Standard Deviation'
filtered_lines = [line for line in lines if 'Average' in line]

results = [float(line.split()[-2]) for line in filtered_lines]

results_per_category = np.array(results).reshape(5, 6, 2)

results_per_category = results_per_category.swapaxes(1, 2).reshape(10, 6)

x_axis = [1, 2, 4, 8, 16, 32]
for i_operation, operation_name in enumerate(["hpe", "hpe_vjp", "opsa", "opsa_vjp", "sap", "sap_vjp", "opsaw", "opsaw_vjp", "sasaw", "sasaw_vjp"]):
    if i_operation % 2 == 0: plt.figure()
    plt.plot(x_axis, x_axis, color="black")
    plt.plot(x_axis, results_per_category[i_operation][0]/results_per_category[i_operation], label=operation_name)
    plt.x_lim = (1, 32)
    plt.y_lim = (1, 32)
    plt.xscale('log')
    plt.yscale('log')
    ticks = [1, 2, 4, 8, 16, 32]
    plt.xticks(x_axis, labels=x_axis)
    plt.yticks(x_axis, labels=x_axis)
    plt.tick_params(axis='x', which='minor', length=0)
    plt.tick_params(axis='y', which='minor', length=0)
    plt.xlabel("Number of threads")
    plt.ylabel("Speed-up")
    plt.legend()
    if i_operation %2 == 1: plt.savefig(f"{operation_name}_strong.pdf")
    if i_operation %2 == 1: plt.close()
