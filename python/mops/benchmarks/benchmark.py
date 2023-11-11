import numpy as np
import time
import matplotlib.pyplot as plt


def benchmark(function, repeats=1000, plot=True):

    timings = []
    for _ in range(repeats):
        start = time.time()
        function()
        end = time.time()
        timings.append(end-start)

    times_array = np.array(timings)
    mean = np.mean(times_array)
    std = np.std(times_array)
    if std > 0.1 * mean: print("warning: inconsistent timings")

    if plot:
        plt.plot(np.arange(times_array.shape[0]), times_array, ".")
        plt.savefig("benchmark_plot.pdf")

    return mean, std
