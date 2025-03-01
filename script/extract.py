import argparse
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import shogi
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

# 定数
RESULT_HEADER = ["sfen", "feature", "opponent_feature", "probability"]

# パラメータ
USER_NAME = "garden0905"
INCORRECT_DATA_NUM = 1
MOVE_NUMBER_LIMIT = 10


def extract_feature(board: shogi.Board, side: str) -> tuple[np.ndarray, np.ndarray]:
    """
    盤面の特徴量を抽出する関数
    Args:
        board (shogi.Board): 盤面情報
        side (str): 評価対象の色
    Returns:
        tuple[np.ndarray, np.ndarray]: 評価対象の特徴量と相手の特徴量
    """
    feature = np.zeros((14, 9, 9), dtype=np.int8)
    opponent_feature = np.zeros((14, 9, 9), dtype=np.int8)

    # 各マスにある駒を特徴量に変換
    for i in range(9):
        for j in range(9):
            if side == shogi.BLACK:
                piece = board.piece_at(i * 9 + j)
            else:
                piece = board.piece_at((8 - i) * 9 + (8 - j))
            if piece is not None:
                if piece.color == side:
                    feature[piece.piece_type - 1, i, j] = 1
                else:
                    opponent_feature[piece.piece_type - 1, i, j] = 1

    return feature, opponent_feature


def gen_correct_data(board: shogi.Board, side: int) -> pd.DataFrame:
    """正解データを生成する関数"""
    feature, opponent_feature = extract_feature(board, side)
    row = {
        "sfen": board.sfen(),
        "feature": feature.tolist(),
        "opponent_feature": opponent_feature.tolist(),
        "probability": 1.0,
    }
    return pd.DataFrame([row])


def gen_incorrect_data(board: shogi.Board, side: int) -> pd.DataFrame:
    """不正解データを生成する関数"""
    rows = []

    # 合法手をランダムに選択
    legal_moves = list(board.legal_moves)
    moves = random.sample(legal_moves, min(INCORRECT_DATA_NUM, len(legal_moves)))
    for move in moves:
        board.push(move)
        feature, opponent_feature = extract_feature(board, side)
        rows.append(
            {
                "sfen": board.sfen(),
                "feature": feature.tolist(),
                "opponent_feature": opponent_feature.tolist(),
                "probability": 0.0,
            }
        )
        board.pop()

    return pd.DataFrame(rows)


def extract_sfen(sfen: str, side: int) -> pd.DataFrame:
    df = pd.DataFrame(columns=RESULT_HEADER)

    # 初期盤面の生成
    board = shogi.Board()
    moves = sfen.split(" ")[7:]

    # 各局面のデータを生成
    for move in moves:
        board.push_usi(move)
        if board.is_game_over() or board.move_number > MOVE_NUMBER_LIMIT:
            break
        if side == board.turn:
            data = gen_incorrect_data(board, side)
        else:
            data = gen_correct_data(board, side)
        df = pd.concat([df, data], axis=0)

    return df


def main(args):

    # データフレームの読み込み
    df = pd.read_csv(args.input)

    # ユーザの絞り込み
    df = df[df["username"] == USER_NAME]

    # 特徴量の抽出
    result_df = pd.DataFrame(columns=RESULT_HEADER)
    df = df.head(5)
    for index, row in tqdm(df.iterrows(), total=len(df)):

        # 局面の抽出
        side = shogi.BLACK if row["side"] == "black" else shogi.WHITE
        data = extract_sfen(row["sfen_body"], side)
        result_df = pd.concat([result_df, data], axis=0)

    # データの保存
    result_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="data/kifu.csv")
    parser.add_argument("-o", "--output", type=str, default="data/extracted.csv")
    args = parser.parse_args()
    main(args)
