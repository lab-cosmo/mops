import math
import time

import matplotlib.pyplot as plt
import numpy as np


def benchmark(function, repeats=1000, warmup=10, plot=True):
    for _ in range(warmup):
        function()

    timings = []
    for _ in range(repeats):
        start = time.time()
        function()
        end = time.time()
        timings.append(end - start)

    times_array = np.array(timings)
    mean = np.mean(times_array)
    std = np.std(times_array)
    if std > 0.1 * mean:
        print("warning: inconsistent timings")

    if plot:
        plt.plot(np.arange(times_array.shape[0]), times_array, ".")
        plt.savefig("benchmark_plot.pdf")

    return mean, std


def format_mean_std(mean, std_dev, decimals=2):
    # find the exponent
    if mean != 0:
        exponent = math.floor(math.log10(abs(mean)))
    else:
        exponent = 0

    # scale the mean and standard deviation by the exponent
    scaled_mean = mean / (10**exponent)
    scaled_std_dev = std_dev / (10**exponent)

    # format the scaled mean and standard deviation
    format_string = f"{{:.{decimals}f}}"
    formatted_mean = format_string.format(scaled_mean)
    formatted_std_dev = format_string.format(scaled_std_dev)
    final_string = f"({formatted_mean}±{formatted_std_dev})e{exponent}"

    return final_string
