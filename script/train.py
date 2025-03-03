"""
train.py

Description:
    棋風を学習するためのモデルを学習するスクリプト

Options:
    -i, --input:  入力データセットのパス
    -o, --output: モデルの保存先のパス
    -e, --epoch:  学習エポック数
"""

import argparse
import ast
import csv
import os
import sys
from pprint import pprint

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm, trange,

# 定数
FEATURE_KEYS = (
    "loc",
    "opp_loc",
)


class ShogiIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size

    def __iter__(self):
        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            loc_batch = []
            opp_loc_batch = []
            probability_batch = []
            for row in reader:
                loc_batch.append(ast.literal_eval(row["loc"]))
                opp_loc_batch.append(ast.literal_eval(row["opp_loc"]))
                probability_batch.append(float(row["probability"]))
                if len(loc_batch) == self.batch_size:
                    yield (
                        torch.tensor(loc_batch, dtype=torch.float32),
                        torch.tensor(opp_loc_batch, dtype=torch.float32),
                        torch.tensor(probability_batch, dtype=torch.float32).view(
                            -1, 1
                        ),
                    )
                    loc_batch = []
                    opp_loc_batch = []
                    probability_batch = []
            if len(loc_batch) > 0:
                yield (
                    torch.tensor(loc_batch, dtype=torch.float32),
                    torch.tensor(opp_loc_batch, dtype=torch.float32),
                    torch.tensor(probability_batch, dtype=torch.float32).view(-1, 1),
                )

    def __len__(self):
        return sum(1 for _ in open(self.file_path))


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

    def forward(self, loc, opp_loc):
        loc = self.loc_convs(loc)
        opp_loc = self.opp_loc_convs(opp_loc)
        x = torch.cat([loc, opp_loc], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        


def main(args):
    # モデルの初期化
    model = ShogiModel().to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # データセットの初期化
    dataset = ShogiIterableDataset(args.input, args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
    
    # 学習
    model.train()
    for epoch in trange(args.epoch):
        total_loss = 0
        for loc, opp_loc, probability in dataloader:
            loc = loc.to(args.device)
            opp_loc = opp_loc.to(args.device)
            probability = probability.to(args.device)

            optimizer.zero_grad()
            output = model(loc, opp_loc)
            loss = criterion(output, probability)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tqdm.write(f"epoch: {epoch}, loss: {total_loss}")


    # 試しに推論
    model.eval()
    output = model(loc, opp_loc)
    pprint(output)

    # モデルの保存
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=str, default="data/extracted.csv")
    parser.add_argument("-o", "--output", type=str, default="model.pth")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
