import hashlib
import numpy as np
from nltk.corpus import movie_reviews


reviews = []
for fileid in movie_reviews.fileids("pos"):
    reviews.extend(movie_reviews.words(fileid))
for fileid in movie_reviews.fileids("neg"):
    reviews.extend(movie_reviews.words(fileid))
ground_truth = len(set(reviews))


### TODO
class FlajoletMartin:
    def __init__(self, num_groups, num_hashes):
        self.num_groups = num_groups
        self.num_hashes = num_hashes
        self.result_matrix = np.zeros((num_groups, num_hashes))

    def trailing_zeros(self, value):
        binary = f"{value:b}"
        return len(binary) - len(binary.rstrip("0"))

    def update(self, item):
        for i in range(self.num_groups):
            for j in range(self.num_hashes):
                salt = f"flajolet_martin_{i}_{j}"
                hasher = hashlib.sha256(salt.encode())
                hasher.update(item.encode())
                result = self.trailing_zeros(int(hasher.hexdigest(), 16))
                self.result_matrix[i, j] = max(self.result_matrix[i, j], result)

    def count(self):
        median = np.median(self.result_matrix, axis=1)
        return int(np.mean(2 ** median))
### end of TODO


if __name__ == "__main__":
    fm = FlajoletMartin(2, 5)
    for word in reviews:
        fm.update(word)
    estimated_result = fm.count()
    print("ground truth:")
    print(f"    {ground_truth}")
    print("estimated result:")
    print(f"    {estimated_result}")
