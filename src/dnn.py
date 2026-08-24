import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible MLP regressor implemented with PyTorch."""

    def __init__(
        self,
        hidden_sizes=(128, 64),
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        optimizer="adam",
        scheduler="none",
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=64,
        epochs=100,
        loss="mse",
        device="cpu",
        random_state=42,
        verbose=False,
    ):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _activation_layer(self, torch):
        activations = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "leaky_relu": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
        }
        if self.activation not in activations:
            raise ValueError(f"Unknown activation: {self.activation}")
        return activations[self.activation]()

    def _build_model(self, torch, input_size):
        layers = []
        previous_size = input_size

        for hidden_size in self.hidden_sizes:
            layers.append(torch.nn.Linear(previous_size, hidden_size))
            if self.batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(self._activation_layer(torch))
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            previous_size = hidden_size

        layers.append(torch.nn.Linear(previous_size, 1))
        return torch.nn.Sequential(*layers)

    def _select_device(self, torch):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, X, y):
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "PyTorch is required for model 'dnn'. Install torch first."
            ) from error

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.y_mean_ = float(y_array.mean())
        self.y_std_ = float(y_array.std()) or 1.0
        y_scaled = (y_array - self.y_mean_) / self.y_std_

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_array),
            torch.from_numpy(y_scaled),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.device_ = self._select_device(torch)
        self.model_ = self._build_model(torch, X_array.shape[1]).to(self.device_)

        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        if self.optimizer not in optimizers:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        optimizer = optimizers[self.optimizer](
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        losses = {
            "mse": torch.nn.MSELoss,
            "mae": torch.nn.L1Loss,
            "huber": torch.nn.HuberLoss,
        }
        if self.loss not in losses:
            raise ValueError(f"Unknown loss: {self.loss}")
        loss_function = losses[self.loss]()

        lr_scheduler = None
        if self.scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
            )
        elif self.scheduler != "none":
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                predictions = self.model_(batch_X)
                batch_loss = loss_function(predictions, batch_y)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item() * len(batch_X)

            if lr_scheduler is not None:
                lr_scheduler.step()

            if self.verbose and (epoch + 1) % 10 == 0:
                mean_loss = epoch_loss / len(dataset)
                print(f"Epoch {epoch + 1}: loss={mean_loss:.5f}")

        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X):
        import torch

        X_array = np.asarray(X, dtype=np.float32)
        tensor = torch.from_numpy(X_array).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(tensor).cpu().numpy().reshape(-1)

        return predictions * self.y_std_ + self.y_mean_