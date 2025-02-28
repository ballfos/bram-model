import argparse
import os
import random
import warnings

import pandas as pd
import shogi

warnings.simplefilter(action="ignore", category=FutureWarning)

USER_NAME = "garden0905"


def extract_feature(board: shogi.Board) -> str:
    return "[abc, hoge]"


def gen_data(board: shogi.Board) -> pd.DataFrame:
    df = pd.DataFrame(columns=["sfen", "feature", "probability"])

    # 現在の局面を正解データとして追加
    sfen = board.sfen()
    feature = extract_feature(board)
    probability = 1.0
    df = pd.concat(
        [df, pd.DataFrame([[sfen, feature, probability]], columns=df.columns)], axis=0
    )

    # 合法手からランダムに不正解データを追加
    legal_moves = list(board.legal_moves)
    moves = random.sample(legal_moves, min(8, len(legal_moves)))
    for move in moves:
        board.push(move)
        sfen = board.sfen()
        feature = extract_feature(board)
        probability = 0.0
        df = pd.concat(
            [df, pd.DataFrame([[sfen, feature, probability]], columns=df.columns)],
            axis=0,
        )
        board.pop()

    return df


def extract_sfen(sfen: str, side: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=["sfen", "feature", "probability"])

    # 初期盤面
    board = shogi.Board()
    moves = sfen.split(" ")[7:]

    for i, move in enumerate(moves):
        board.push_usi(move)
        if side == "black" and i % 2 == 1:
            continue
        if side == "white" and i % 2 == 0:
            continue

        data = gen_data(board)
        df = pd.concat([df, data], axis=0)

    return df


def main(args):

    # データフレームの読み込み
    df = pd.read_csv(args.input)

    # ユーザの絞り込み
    df = df[df["username"] == USER_NAME]

    # 特徴量の抽出
    result_df = pd.DataFrame(columns=["sfen", "feature", "probability"])
    for index, row in df.head(8).iterrows():
        # 進捗表示
        print(f"{index+1}/{len(df)}")

        # 局面の抽出
        data = extract_sfen(row["sfen_body"], row["side"])
        result_df = pd.concat([result_df, data], axis=0)

    # データの保存
    result_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="data/kifu.csv")
    parser.add_argument("-o", "--output", type=str, default="data/extracted.csv")
    args = parser.parse_args()
    main(args)
