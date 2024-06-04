import csv
import numpy as np
from argparse import ArgumentParser


class KMeans:
    def __init__(self, n_dim: int, n_clusters: int):
        self.n_dim = n_dim
        self.n_clusters = n_clusters

    def _init_centroids_random(self, X: np.ndarray):
        """Randomly initialize centroids.

        Args:
            X (np.ndarray): Input data of shape (N, D)
        """
        N, _ = X.shape
        indices = np.random.choice(N, self.n_clusters, replace=False)
        return X[indices]

    def _init_centroids_dispersed_means(self, X: np.ndarray):
        """Initialize centroids by dispersed means.

        Args:
            X (np.ndarray): Input data of shape (N, D)
        """
        N, _ = X.shape
        choices = np.zeros(self.n_clusters, dtype=int)
        choices[0] = np.random.randint(N)
        dist2means = np.full(N, np.infty)
        for i in range(self.n_clusters - 1):
            for candidate in range(N):
                dist2means[candidate] = min(
                    dist2means[candidate], np.linalg.norm(X[candidate] - X[choices[i]])
                )
            choices[i + 1] = np.argmax(dist2means)

        return X[choices]

    def init_centroids(self, X: np.ndarray, init: str) -> np.ndarray:
        """Initialize centroids by randomly selecting samples or dispersed means.

        Args:
            X (np.ndarray): Input data of shape (N, D)
            init (str): Initialization method. Either "random" or "dispersed_means".

        Returns:
            np.ndarray: Initial centroids of shape (n_clusters, n_dim)
        """

        if init == "random":
            return self._init_centroids_random(X)
        elif init == "dispersed_means":
            return self._init_centroids_dispersed_means(X)
        else:
            raise ValueError(f"Invalid method: {init}")

    def _remap_labels(self, radius: np.ndarray, labels: np.ndarray):
        new_labels = np.zeros(self.n_clusters, dtype=int)
        indices = np.argsort(radius)
        new_labels[indices] = np.arange(self.n_clusters)

        return new_labels[labels]

    def calc_radius(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        """Calculate the radius of each cluster.

        Args:
            X (np.ndarray): Input data of shape (N, D).
            labels (np.ndarray): An array of shape (N,). The cluster labels for each sample.
            centroids (np.ndarray): An array of shape (n_clusters, n_dim). The cluster centroids.

        Returns:
            np.ndarray: An array of shape (n_clusters,). The radius of each cluster.
        """
        # TODO: Compute the radius of each cluster.
        # The radius of a cluster is defined as the maximum l2-distance
        # between the centroid and any sample in the cluster.
        radius = np.zeros(self.n_clusters, dtype=float)
        for i in range(self.n_clusters):
            candidates = X[labels == i]
            radius[i] = np.max(np.linalg.norm(candidates - centroids[i], axis=1))
        # End of TODO

        return radius

    def cluster(
        self,
        X: np.ndarray,
        init: str,
        max_iter: int = 100,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """KMeans clustering algorithm.

        Args:
            X (np.ndarray): Input data of shape (N, D)
            init (str): Initialization. Either "random" or "dispersed_means".
            max_iter (int): Maximum number of iterations. Defaults to 100.
            eps (float): Tolerance for early stopping. Defaults to 1e-6.

        Returns:
            np.ndarray: An array of shape (N,). The cluster labels for each sample.
        """
        N, D = X.shape
        centroids = self.init_centroids(X, init)
        labels = np.zeros((N), dtype=int)

        # TODO: Implement the KMeans algorithm here.
        # You are free to add any helper methods and/or attributes if needed.
        # However, DO NOT delete any parameter in function signatures.
        for _ in range(max_iter):
            for i in range(N):
                labels[i] = np.argmin(np.linalg.norm(X[i] - centroids, axis=1))
            new_centroids = np.zeros_like(centroids)
            for i in range(self.n_clusters):
                new_centroids[i] = np.mean(X[labels == i], axis=0)
            if np.linalg.norm(new_centroids - centroids) < eps:
                break
            centroids = new_centroids
        # End of TODO
        # NOTE: Remember to return the labels, which should be an array of shape (N,),
        # where each element indicates the cluster index of the corresponding sample.

        radius = self.calc_radius(X, labels, centroids)
        labels = self._remap_labels(radius, labels)
        print(radius)

        return labels


def load_data(data_path: str = "./data/features.csv"):
    """
    Helper function to load data. Returns an array of shape (N, D).
    DO NOT modify this function.
    """
    data = []
    with open(data_path, "r", encoding="utf-8") as fi:
        fi.readline()  # skip header
        csv_reader = csv.reader(fi)
        paper_id = 0
        for row in csv_reader:
            data.append(np.array(row[1:], dtype=float))
            paper_id += 1

    return np.array(data)


def save_result(filename: str, labels: np.ndarray):
    """Helper function to save results to a CSV file.
    DO NOT modify this function.

    Args:
        filename (str): The output file path.
        results (np.ndarray): An array of shape (N,). The cluster labels for each sample.
    """
    with open(filename, "w", encoding="utf-8") as fo:
        fo.write("id,category\n")
        for i, result in enumerate(labels):
            fo.write(f"{i},{result}\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/features.csv")
    parser.add_argument("--output", type=str, default="./data/predictions.csv")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--init",
        choices=["random", "dispersed_means"],
        default="dispersed_means",
    )

    return parser.parse_args()


def main():
    # Driver code. DO NOT modify.
    args = parse_args()
    n_clusters = 5
    n_dims = 100
    n_samples = 50000

    X = load_data(args.data)
    assert X.shape == (n_samples, n_dims)

    kmeans = KMeans(n_dims, n_clusters)

    clusters = kmeans.cluster(X, args.init, args.max_iter, args.eps)

    save_result(args.output, clusters)


if __name__ == "__main__":
    main()
