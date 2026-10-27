import numpy as np
from scipy import sparse
from scipy.linalg import solve


class ImplicitALS:
    def __init__(self, factors=32, alpha=20, regularization=1.0, iterations=10, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def fit(self, train_pairs):
        self.users = sorted(train_pairs["user_id"].unique())
        self.items = np.array(sorted(train_pairs["item_id"].unique()))
        self.user_index = {user: i for i, user in enumerate(self.users)}
        self.item_index = {item: i for i, item in enumerate(self.items)}
        rows = train_pairs["user_id"].map(self.user_index).to_numpy()
        cols = train_pairs["item_id"].map(self.item_index).to_numpy()
        preferences = self._preferences(train_pairs)
        interactions = sparse.csr_matrix(
            (preferences, (rows, cols)), shape=(len(self.users), len(self.items))
        )
        item_interactions = interactions.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (len(self.items), self.factors))

        for _ in range(self.iterations):
            self.user_factors = self._update(interactions, self.item_factors)
            self.item_factors = self._update(item_interactions, self.user_factors)
        return self

    def _preferences(self, train_pairs):
        return train_pairs["rating"].ge(4).to_numpy(dtype=float)

    def _update(self, interactions, fixed_factors):
        gram = fixed_factors.T @ fixed_factors
        base = gram + self.regularization * np.eye(self.factors)
        updated = np.empty((interactions.shape[0], self.factors))

        for row in range(interactions.shape[0]):
            left, right = interactions.indptr[row:row + 2]
            indices = interactions.indices[left:right]
            preferences = interactions.data[left:right]
            neighbors = fixed_factors[indices]
            matrix = base + self.alpha * (neighbors.T @ neighbors)
            target = (1 + self.alpha) * (neighbors.T @ preferences)
            updated[row] = solve(matrix, target, assume_a="pos", check_finite=False)
        return updated

    def recommend(self, user_ids, seen, k=10):
        recommendations = {}
        for start in range(0, len(user_ids), 256):
            batch = user_ids[start:start + 256]
            rows = [self.user_index[user] for user in batch]
            scores = self.user_factors[rows] @ self.item_factors.T
            for row, user in enumerate(batch):
                excluded = [self.item_index[item] for item in seen[user] if item in self.item_index]
                scores[row, excluded] = -np.inf
            top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for user, indices in zip(batch, top):
                recommendations[user] = self.items[indices].tolist()
        return recommendations
