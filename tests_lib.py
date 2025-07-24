import matplotlib.pyplot as plt
import numpy as np


def pdf_comparison(q, title=None):
    plt.plot(q['cp_quantiles'], q['cp_pdf'], label='cp', color='red', linewidth=1)
    plt.plot(q['ml_quantiles'], q['ml_pdf'], label='ml', color='blue', linewidth=1)
    plt.xlabel('q')
    plt.ylabel('pdf')
    plt.legend()
    plt.title(title)
    plt.show()


def cdf_comparison(q, p, edf=False, title=None):
    # plot cp, ml comparison
    if edf:
        plt.plot(q['cp_quantiles'], 1-p, label='cp', color='red', linewidth=1)
        plt.plot(q['ml_quantiles'], 1-p, label='ml', color='blue', linewidth=1)
    else:
        plt.plot(q['cp_quantiles'], p, label='cp', color='red', linewidth=1)
        plt.plot(q['ml_quantiles'], p, label='ml', color='blue', linewidth=1)
    plt.axhline(y=1, color='black', linewidth=1)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.legend()
    plt.title(title)
    plt.show()


def empirical(x):
    # plot the empirical distribution
    cdf = np.zeros(len(x))
    for i in range(len(x)):
        cdf[i] = np.sum(x <= x[i]) / (len(x)+1)
    plt.scatter(x, 1-cdf, label='empirical', color='black', marker='x')