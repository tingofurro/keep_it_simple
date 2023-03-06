import os
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
from matplotlib import pyplot as plt
from zca import ZCA

sns.set(style="ticks", context="paper")
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["figure.dpi"] = 300


def str_list_to_real_list(text: str) -> List:
    return [float(d) for d in text.strip("][").replace('"', "").split(",")]


def make_density(reward_component_data, subplot_color, subplot_ax, title):
    """
    Function to plot the histogram for a single list.
    """
    sns.histplot(
        reward_component_data,
        stat="probability",
        kde=True,
        color=subplot_color,
        ax=subplot_ax,
    )
    subplot_ax.set_xlim(0, 1.00)
    if title == "Fluency":
        subplot_ax.set_ylim(ymax=1.00)

    # Add mode
    dataframe = pd.DataFrame(reward_component_data)
    reward_component_data_mode = dataframe.median()[0]
    subplot_ax.axvline(
        reward_component_data_mode,
        color=subplot_color,
        linestyle="dashed",
        linewidth=1.5,
    )

    reward_component_data_min = dataframe.min()[0]
    reward_component_data_max = dataframe.max()[0]

    subplot_ax.set_title(title)

    text = "Min: {:.4f}\nMedian: {:.4f}\nMax: {:.4f}".format(
        reward_component_data_min, reward_component_data_mode, reward_component_data_max
    )
    props = dict(boxstyle="round", facecolor=subplot_color, alpha=0.5)
    subplot_ax.text(
        0.6,
        0.99,
        text,
        transform=subplot_ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=props,
    )


def make_density_whitening(reward_component_data, subplot_color, subplot_ax, title):
    """
    Function to plot the histogram for a single list.
    """
    X = np.array([reward_component_data])
    trf = ZCA().fit(X.T)
    X_whitened = trf.transform(X.T)
    transpose_X_whitened = list(X_whitened.T[0])
    sns.histplot(
        transpose_X_whitened,
        stat="probability",
        kde=True,
        color=subplot_color,
        ax=subplot_ax,
    )
    if title == "Fluency":
        subplot_ax.set_ylim(ymax=1.00)

    # Add mode
    dataframe = pd.DataFrame(transpose_X_whitened)
    reward_component_data_mode = dataframe.median()[0]
    subplot_ax.axvline(
        reward_component_data_mode,
        color=subplot_color,
        linestyle="dashed",
        linewidth=1.5,
    )

    reward_component_data_min = dataframe.min()[0]
    reward_component_data_max = dataframe.max()[0]

    subplot_ax.set_title(title)

    text = "Min: {:.4f}\nMedian: {:.4f}\nMax: {:.4f}".format(
        reward_component_data_min, reward_component_data_mode, reward_component_data_max
    )
    props = dict(boxstyle="round", facecolor=subplot_color, alpha=0.5)
    subplot_ax.text(
        0.6,
        0.99,
        text,
        transform=subplot_ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=props,
    )


root_dir = "results"
csv_files_name = os.listdir(root_dir)

# All data
coverage_scores = []
simple_syn_scores = []
simple_lex_scores = []
fluency_lm_scores = []
fluency_disc_scores = []
for csv_file_name in csv_files_name:
    if "guardrails" not in csv_file_name and ".gitkeep" not in csv_file_name:
        print("Importing data of experiments: ", csv_file_name)
        data = pd.read_csv(os.path.join(root_dir, csv_file_name))
        subset_data = data[:40000]

        for row in subset_data.iterrows():
            coverage_scores.extend(str_list_to_real_list(row[1][0]))
            simple_syn_scores.extend(str_list_to_real_list(row[1][1]))
            simple_lex_scores.extend(str_list_to_real_list(row[1][2]))
            fluency_lm_scores.extend(str_list_to_real_list(row[1][3]))
            fluency_disc_scores.extend(str_list_to_real_list(row[1][4]))

all_stat_list = [
    coverage_scores,
    simple_syn_scores,
    simple_lex_scores,
    fluency_lm_scores,
    fluency_disc_scores,
]
num_subplots = len(all_stat_list)
n_cols = 3
n_rows = ceil(num_subplots / n_cols)

fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 3))
colors = plt.cm.tab10.colors
labels = [
    "Coverage",
    "Syntactic Simplicity",
    "Lexical Simplicity",
    "Fluency",
    "Fluency Discriminator",
]
for ax, stat, color, label in zip(np.ravel(axes), all_stat_list, colors, labels):
    make_density(stat, color, ax, title=label)

for ax in np.ravel(axes)[
    num_subplots:
]:  # To remove possible empty subplots at the end.
    ax.remove()

fig.suptitle("All data", fontsize=12)
plt.tight_layout()
plt.savefig("all_data.png")
plt.show()

# Last 10K data
coverage_scores = []
simple_syn_scores = []
simple_lex_scores = []
fluency_lm_scores = []
fluency_disc_scores = []
for csv_file_name in csv_files_name:
    if "guardrails" not in csv_file_name and ".gitkeep" not in csv_file_name:
        print("Importing data of experiments: ", csv_file_name)
        data = pd.read_csv(os.path.join(root_dir, csv_file_name))
        subset_data = data[30000:40000]

        for row in subset_data.iterrows():
            coverage_scores.extend(str_list_to_real_list(row[1][0]))
            simple_syn_scores.extend(str_list_to_real_list(row[1][1]))
            simple_lex_scores.extend(str_list_to_real_list(row[1][2]))
            fluency_lm_scores.extend(str_list_to_real_list(row[1][3]))
            fluency_disc_scores.extend(str_list_to_real_list(row[1][4]))

tenk_stat_list = [
    coverage_scores,
    simple_syn_scores,
    simple_lex_scores,
    fluency_lm_scores,
    fluency_disc_scores,
]
num_subplots = len(tenk_stat_list)
n_cols = 3
n_rows = ceil(num_subplots / n_cols)

fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 3))

for ax, stat, color, label in zip(np.ravel(axes), tenk_stat_list, colors, labels):
    make_density(stat, color, ax, title=label)

for ax in np.ravel(axes)[
    num_subplots:
]:  # To remove possible empty subplots at the end.
    ax.remove()

fig.suptitle("Last 10K", fontsize=16)
plt.tight_layout()
plt.savefig("last_10k.png")
plt.show()

# Whitened data
# Documentation:
# https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
# https://arxiv.org/abs/2103.15316
# https://github.com/mwv/zca

num_subplots = len(all_stat_list)
n_cols = 3
n_rows = ceil(num_subplots / n_cols)

fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 3))

for ax, stat, color, label in zip(np.ravel(axes), all_stat_list, colors, labels):
    make_density_whitening(stat, color, ax, title=label)

for ax in np.ravel(axes)[
    num_subplots:
]:  # To remove possible empty subplots at the end.
    ax.remove()

fig.suptitle("Whitened All Data", fontsize=16)
plt.tight_layout()
plt.savefig("all_data_whitened.png")
plt.show()
