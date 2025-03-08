"""
train.py

Description:
    棋風を学習するためのモデルを学習するスクリプト

Options:
    -i, --input:  入力データセットのパス
    -o, --output: モデルの保存先のパス
    -d, --device: 学習に使用するデバイス (cpu | cuda)
    -e, --epoch:  学習エポック数
    -b, --batch_size: バッチサイズ
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(__file__))
from model import ShogiModel

# 定数
FEATURE_KEYS = (
    "loc",
    "opp_loc",
)


class ShogiDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, chunk_size=8192):
        self.data = []
        lines = sum(1 for _ in open(file_path))
        for chunk in tqdm(
            pd.read_csv(
                file_path, usecols=("probability",) + FEATURE_KEYS, chunksize=chunk_size
            ),
            total=lines // chunk_size + 1,
        ):
            for key in FEATURE_KEYS:
                chunk[key] = chunk[key].apply(json.loads)
            self.data.append(chunk)
        self.data = pd.concat(self.data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            torch.tensor(row["loc"], dtype=torch.float32),
            torch.tensor(row["opp_loc"], dtype=torch.float32),
            torch.tensor(row["probability"], dtype=torch.float32),
        )


def main(args):
    # モデルの初期化
    print("Initializing model...")
    model = ShogiModel().to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # データセットの初期化
    print("Loading dataset...")
    dataset = ShogiDataset(args.input)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # 学習
    print("Training model...")
    model.train()
    for epoch in trange(args.epoch):
        total_loss = 0
        for loc, opp_loc, probability in dataloader:
            loc = loc.to(args.device)
            opp_loc = opp_loc.to(args.device)
            probability = probability.to(args.device).view(-1, 1)

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
    for i in range(10):
        print(f"probability: {probability[i].item()}, output: {output[i].item()}")

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
