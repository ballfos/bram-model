import torch
from torch import nn


class ShogiModel(nn.Module):
    def __init__(self):
        super(ShogiModel, self).__init__()
        self.loc_convs = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.opp_loc_convs = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 9 * 9 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, loc, opp_loc):
        loc = self.loc_convs(loc)
        opp_loc = self.opp_loc_convs(opp_loc)
        x = torch.cat([loc, opp_loc], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
