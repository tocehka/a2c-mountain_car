import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, actions_dim, states_dim):
        super().__init__()
        self.fc1 = nn.Linear(states_dim, 30)
        self.fc2 = nn.Linear(30, 50)
        self.fc3 = nn.Linear(50, actions_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)


class Critic(nn.Module):
    def __init__(self, states_dim):
        super().__init__()
        self.fc1 = nn.Linear(states_dim, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x