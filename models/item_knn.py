import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors

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

        norms = np.sqrt(self.interactions.power(2).sum(axis=0)).A1
        inverse_norms = np.zeros_like(norms)
        np.divide(1, norms, out=inverse_norms, where=norms > 0)
        normalized = (self.interactions @ sparse.diags(inverse_norms)).tocsr()

        # Считаем сходства блоками, не создавая полную плотную матрицу.
        indices, values = [], []
        indptr = [0]
        for start in range(0, len(self.items), 256):
            similarities = (normalized[:, start:start + 256].T @ normalized).tocsr()
            for row in range(similarities.shape[0]):
                left, right = similarities.indptr[row:row + 2]
                neighbors = similarities.indices[left:right]
                scores = similarities.data[left:right]
                keep = neighbors != start + row
                neighbors, scores = neighbors[keep], scores[keep]
                order = np.lexsort((neighbors, -scores))[:self.n_neighbors]
                indices.extend(neighbors[order])
                values.extend(scores[order])
                indptr.append(len(indices))

        self.similarity = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), indices, indptr),
            shape=(len(self.items), len(self.items)),
        )
        return self

    def recommend(self, user_ids, seen, k=10):
        recommendations = {}
        for start in range(0, len(user_ids), 256):
            batch = user_ids[start:start + 256]
            rows = [self.user_index[user] for user in batch]
            scores = (self.interactions[rows] @ self.similarity.T).toarray()

            for row, user in enumerate(batch):
                excluded = [self.item_index[item] for item in seen[user] if item in self.item_index]
                scores[row, excluded] = -np.inf

            top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for user, item_indices in zip(batch, top):
                recommendations[user] = self.items[item_indices].tolist()
        return recommendations
