"""
preprocess.py

Description:
    収集した CSV データから必要な情報を抽出する前処理スクリプト

Options:
    -i, --input:  入力ファイルのパス
    -o, --output: 出力ファイルのパス
"""

import argparse
import csv
import os

import pandas as pd

TARGET_USERNAME = "garden0905"


def main(args):
    # データの読み込み
    df = pd.read_csv(args.input)

    # データの前処理
    df = df[df["username"] == TARGET_USERNAME]
    df = df[["side", "sfen_body"]].rename(columns={"sfen_body": "sfen"})

    # データの保存
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="data/kifu.csv")
    parser.add_argument("-o", "--output", type=str, default="data/preprocessed.csv")
    args = parser.parse_args()
    main(args)
