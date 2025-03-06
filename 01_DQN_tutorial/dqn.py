"""
    一个由全连接层构成的DQN
    输入为state，输出为Q(s),在这个
"""
import torch
from torch import nn
from torch.nn import functional as F
class DQN(nn.Module):
    #n_observations = 4 n_actions = 2
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

if __name__ == "__main__":
    net = DQN(4,2)
    state_batch = torch.randn(32, 4)
    out = net(state_batch)
    print(out)