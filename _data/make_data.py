from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(palette="tab10")


N_CLUSTERS = 4      # How many clusters are there?
CLUSTER_SCALE = 25  # How spread out should they be?
CLUSTER_SIZE = 300  # How many points in each cluster?

FEATURES_FILE = Path("./data.csv")
TARGETS_FILE = Path("./targets.csv")
PLOT_NAME = Path("./ground-truth-plot.png")

def get_centers(n: int, scale: float):
    return np.random.random((n, 2)) * scale

def make_cluster(center: List[float], n: int):
    return np.random.normal(size=(n,2)) + center

def get_targets(n_clusters: int, cluster_size: int):
    return np.array(list(chain(*([i for _ in range(cluster_size)] for i in range(n_clusters)))))


def save_features(data: np.ndarray, path: Path):
    sdata = "\n".join(",".join(map(str,row)) for row in data.tolist())
    Path(path).write_text(sdata)

def save_targets(data: np.ndarray, path: Path):
    sdata = "\n".join(map(str, data.tolist()))
    Path(path).write_text(sdata)


def main():
    # Generate the data
    centers = get_centers(N_CLUSTERS, CLUSTER_SCALE)
    clusters = np.concatenate([make_cluster(c, CLUSTER_SIZE) for c in centers])
    targets = get_targets(N_CLUSTERS, CLUSTER_SIZE)

    # Display the data
    colors = np.array(sns.color_palette())
    plt.scatter(*zip(*clusters), c=colors[targets])
    plt.savefig(PLOT_NAME)
    plt.show()

    # Save the data
    save_features(clusters, FEATURES_FILE)
    save_targets(targets, TARGETS_FILE)


if __name__ == '__main__':
    main()    
