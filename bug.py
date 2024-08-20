import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import sparsimony_sparsity as sp


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def objective(trial):
    # Suggest the number of units in the hidden layers
    hidden_size1 = trial.suggest_int("hidden_size1", 50, 400)
    hidden_size2 = trial.suggest_int("hidden_size2", 50, 400)
    # Load your data
    # For demonstration purposes, using random data here
    input_size = 120
    output_size = 24
    X_train = torch.randn(3500, input_size, device="cuda")
    y_train = torch.randn(3500, output_size, device="cuda")

    # Initialize the model
    model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)
    model.to(device="cuda")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    sparsity_level = 0.85
    sparsimony_sparsity = sp.SparsimonySparsity(
        model, optimizer, sparsity_level
    )
    sparsifier = sparsimony_sparsity.sparse_optimizer()
    print(model)

    # Training loop
    n_epochs = 200
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        sparsifier.step()

    # Validation (or you can use a validation set)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        val_loss = criterion(y_pred, y_train).item()
    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial

print(" Value: {}".format(trial.value))

print(" Params: ")
for key, value in trial.params.items():
    print(" {}: {}".format(key, value))
