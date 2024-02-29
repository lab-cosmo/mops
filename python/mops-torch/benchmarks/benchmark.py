import argparse
import gc
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def initialize():
    torch.manual_seed(0xDEADBEEF)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    print(
        f"Running on {parser.parse_args().device} with dtype {parser.parse_args().dtype}"
    )
    torch.set_default_device(parser.parse_args().device)
    torch.set_default_dtype(getattr(torch, parser.parse_args().dtype))


def benchmark(function, repeats=1000, warmup=10, plot=True):
    gc.disable()

    for _ in range(warmup):
        result = function()
        result.backward()

    timings_fwd = []
    timings_bwd = []
    for _ in range(repeats):
        # forward pass
        start = time.time()
        result = function()
        end = time.time()
        timings_fwd.append(end - start)
        # backward pass
        start = time.time()
        result.backward()
        end = time.time()
        timings_bwd.append(end - start)

    times_array_fwd = np.array(timings_fwd)
    mean_fwd = np.mean(times_array_fwd)
    std_fwd = np.std(times_array_fwd)
    if std_fwd > 0.1 * mean_fwd:
        print("warning: inconsistent timings in forward pass")

    times_array_bwd = np.array(timings_bwd)
    mean_bwd = np.mean(times_array_bwd)
    std_bwd = np.std(times_array_bwd)
    if std_bwd > 0.1 * mean_bwd:
        print("warning: inconsistent timings in backward pass")

    if plot:
        plt.plot(
            np.arange(times_array_fwd.shape[0]), times_array_fwd, ".", label="forward"
        )
        plt.plot(
            np.arange(times_array_bwd.shape[0]), times_array_bwd, ".", label="backward"
        )
        plt.legend()
        plt.savefig("benchmark_plot.pdf")

    gc.enable()
    return mean_fwd, std_fwd, mean_bwd, std_bwd


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
