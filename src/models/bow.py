"""Bag of Words model """

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer


class NbowModel(nn.Module):
    """Neural Bag of Words model"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    def __init__(self, vocab_sz):
        """Instantiate the model"""
        super().__init__()
        self.vocab_sz = vocab_sz
        # Instantiate the CountVectorizer
        self.cv = CountVectorizer(
            min_df=0.005,
            max_df=0.75,
            stop_words="english",
            strip_accents="ascii",
            max_features=self.vocab_sz,
        )

        # Define the PyTorch model
        self.dropout = nn.Dropout(0.10)
        self.dense1 = nn.Linear(self.vocab_sz, 15)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()
        self.l1 = 1e-5
        self.l2 = 1e-4

    def forward(self, x):
        """Forward pass"""
        x = self.dropout(x)
        x = self.dense1(x)
        l1_penalty = self.l1 * torch.norm(self.dense1.weight, 1)
        l2_penalty = self.l2 * torch.norm(self.dense1.weight, 2)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x, l1_penalty, l2_penalty

    def fit(
        self, x, y, epochs=10, batch_size=32, validation_split=0.2, lr=0.002
    ):
        """Fit the model"""
        res = self.cv.fit_transform(x).toarray()
        dataset = TensorDataset(
            torch.tensor(res, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs, l1_penalty, l2_penalty = self.forward(inputs)
                loss = (
                    criterion(outputs.squeeze(), labels)
                    + l1_penalty
                    + l2_penalty
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs, _, _ = self.forward(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{epochs} |"
                f"Train loss: {train_loss/len(train_loader):.4f} - "
                f"Val loss: {val_loss/len(val_loader):.4f}"
            )

    def predict(self, x):
        """Predict the class probabilities of x"""
        res = self.cv.transform(x).toarray()
        self.eval()
        with torch.no_grad():
            outputs, _, _ = self.forward(torch.tensor(res, dtype=torch.float32))
        return outputs.squeeze().numpy()

    def eval_acc(self, x, labels, threshold=0.5):
        """Evaluate the accuracy of the model"""
        return accuracy_score(labels, self.predict(x) > threshold)

    def eval_rocauc(self, x, labels):
        """Evaluate the ROC AUC of the model"""
        return roc_auc_score(labels, self.predict(x))

    @property
    def model_dict(self):
        """Get model dictionary"""
        return {"vectorizer": self.cv, "model_state_dict": self.state_dict()}

    @classmethod
    def from_dict(cls, model_dict):
        """Load model from dictionary"""
        nbow_model = cls(len(model_dict["vectorizer"].vocabulary_))
        nbow_model.load_state_dict(model_dict["model_state_dict"])
        nbow_model.cv = model_dict["vectorizer"]
        return nbow_model
