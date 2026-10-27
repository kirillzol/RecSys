from .ials import ImplicitALS


class CustomALS(ImplicitALS):
    def _preferences(self, train_pairs):
        return (train_pairs["rating"].to_numpy(dtype=float) - 3) / 2
