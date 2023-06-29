import torch
import numpy as np
import matplotlib.pyplot as plt


def generate(n=10000, s=1, t=1, k=1):
    x_value = s * torch.randn(n, 1)
    y_value = k * x_value + s * torch.randn(n, 1)
    x_prime = y_value + t * torch.randn(n, 1)
    return torch.cat([x_value, x_prime], dim=1), y_value


def draw_dataset(x_data, y_data, x_label="x", y_label="y", filename='./figure/dataset.png'):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, s=0.5, alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(filename)
    plt.show()


def draw_regression(x_data, y_data, slope, x_label="x", y_label="y", filename='./figure/regression.png'):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, s=0.5, alpha=0.8)
    plt.plot(x_data, slope * x_data, c="red", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(filename)
    plt.show()


def draw_curve(y_data, x_label="epoch", y_label="loss", filename='./figure/curve.png'):
    plt.figure(figsize=(8, 6))
    x_data = range(len(y_data))
    x_data, y_data = np.array(x_data), np.array(y_data)
    plt.plot(x_data, y_data, c="purple", linewidth=1)
    plt.loglog()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(filename)
    plt.show()
