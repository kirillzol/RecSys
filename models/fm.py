import numpy as np
import torch
from torch import nn


class FMNetwork(nn.Module):
    def __init__(self, feature_count, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(feature_count, 1, padding_idx=0)
        self.embeddings = nn.Embedding(feature_count, factors, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embeddings.weight, std=0.01)
        with torch.no_grad():
            self.embeddings.weight[0].zero_()

    def forward(self, features):
        vectors = self.embeddings(features)
        interactions = 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.linear(features).sum(dim=(1, 2)) + interactions


class FactorizationMachine:
    def __init__(self, factors=16, use_metadata=True, epochs=10, learning_rate=0.003, regularization=1e-5, batch_size=2048, seed=42):
        self.factors = factors
        self.use_metadata = use_metadata
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, train_pairs, items):
        self.users = sorted(train_pairs["user_id"].unique())
        self.items = np.array(sorted(train_pairs["item_id"].unique()))
        self.user_index = {user: i for i, user in enumerate(self.users)}
        self.item_index = {item: i for i, item in enumerate(self.items)}
        feature_count = 1 + len(self.users) + len(self.items)
        metadata_index = {}
        metadata = items.set_index("item_id")
        item_features = []
        for i, item in enumerate(self.items):
            features = [1 + len(self.users) + i]
            if self.use_metadata:
                row = metadata.loc[item]
                tags = [("brand", row["brand"])]
                tags += [("category", category) for category in sorted(set(row["categories"]))
                         if category != "Musical Instruments"]
                for tag in tags:
                    if tag not in metadata_index:
                        metadata_index[tag] = feature_count
                        feature_count += 1
                    features.append(metadata_index[tag])
            item_features.append(features)

        width = max(map(len, item_features))
        self.item_features = torch.zeros((len(self.items), width), dtype=torch.long)
        for i, features in enumerate(item_features):
            self.item_features[i, :len(features)] = torch.tensor(features)

        users = train_pairs["user_id"].map(self.user_index).to_numpy()
        item_ids = train_pairs["item_id"].map(self.item_index).to_numpy()
        targets = train_pairs["rating"].ge(4).to_numpy(dtype=np.float32)
        known_pairs = users * len(self.items) + item_ids
        positive_users = users[targets == 1]
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.network = FMNetwork(feature_count, self.factors)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        loss_function = nn.BCEWithLogitsLoss()
        self.loss_history = []

        for _ in range(self.epochs):
            negative_items = rng.integers(len(self.items), size=len(positive_users))
            invalid = np.isin(positive_users * len(self.items) + negative_items, known_pairs)
            while invalid.any():
                negative_items[invalid] = rng.integers(len(self.items), size=invalid.sum())
                invalid = np.isin(positive_users * len(self.items) + negative_items, known_pairs)

            epoch_users = np.concatenate([users, positive_users])
            epoch_items = np.concatenate([item_ids, negative_items])
            epoch_targets = np.concatenate([targets, np.zeros(len(positive_users), dtype=np.float32)])
            order = rng.permutation(len(epoch_users))
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                user_features = torch.from_numpy(epoch_users[batch] + 1).unsqueeze(1)
                features = torch.cat([user_features, self.item_features[epoch_items[batch]]], dim=1)
                labels = torch.from_numpy(epoch_targets[batch])
                optimizer.zero_grad()
                logits = self.network(features)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.loss_history.append(total_loss / len(order))
        return self

    def recommend(self, user_ids, seen, k=10):
        with torch.no_grad():
            vectors = self.network.embeddings(self.item_features)
            item_vectors = vectors.sum(dim=1)
            item_bias = self.network.bias + self.network.linear(self.item_features).sum(dim=(1, 2))
            item_bias += 0.5 * (item_vectors.square() - vectors.square().sum(dim=1)).sum(dim=1)
            recommendations = {}
            for start in range(0, len(user_ids), 256):
                batch = user_ids[start:start + 256]
                user_features = torch.tensor([1 + self.user_index[user] for user in batch])
                scores = self.network.embeddings(user_features) @ item_vectors.T
                scores += item_bias + self.network.linear(user_features)
                scores = scores.numpy()
                for row, user in enumerate(batch):
                    excluded = [self.item_index[item] for item in seen[user] if item in self.item_index]
                    scores[row, excluded] = -np.inf
                top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
                for user, indices in zip(batch, top):
                    recommendations[user] = self.items[indices].tolist()
        return recommendations
