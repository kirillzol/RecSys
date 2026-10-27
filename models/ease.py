import numpy as np
from scipy import sparse
from scipy.linalg import solve


class EASE:
    def __init__(self, regularization=500):
        self.regularization = regularization

    def fit(self, train_pairs):
        self.users = sorted(train_pairs["user_id"].unique())
        self.items = np.array(sorted(train_pairs["item_id"].unique()))
        self.user_index = {user: i for i, user in enumerate(self.users)}
        self.item_index = {item: i for i, item in enumerate(self.items)}
        positive = train_pairs.loc[train_pairs["rating"] >= 4]
        rows = positive["user_id"].map(self.user_index).to_numpy()
        cols = positive["item_id"].map(self.item_index).to_numpy()
        self.interactions = sparse.csr_matrix(
            (np.ones(len(positive), dtype=np.float32), (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )
        self.interactions.data[:] = 1

        gram = (self.interactions.T @ self.interactions).toarray(order="F")
        gram.flat[::len(self.items) + 1] += self.regularization
        identity = np.eye(len(self.items), dtype=np.float32, order="F")
        inverse = solve(
            gram, identity, assume_a="pos", check_finite=False,
            overwrite_a=True, overwrite_b=True,
        ).copy(order="F")
        diagonal = inverse.diagonal().copy()
        inverse /= -diagonal[None, :]
        np.fill_diagonal(inverse, 0)
        self.weights = inverse
        return self

    def recommend(self, user_ids, seen, k=10):
        recommendations = {}
        for start in range(0, len(user_ids), 256):
            batch = user_ids[start:start + 256]
            rows = [self.user_index[user] for user in batch]
            scores = self.interactions[rows] @ self.weights
            for row, user in enumerate(batch):
                excluded = [self.item_index[item] for item in seen[user] if item in self.item_index]
                scores[row, excluded] = -np.inf
            top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for user, indices in zip(batch, top):
                recommendations[user] = self.items[indices].tolist()
        return recommendations
