"""
extract.py

Description:
    将棋の棋風を学習するためのデータセットを生成するスクリプト

Options:
    -i, --input:  入力データセットのパス
    -o, --output: 出力データセットのパス
    -n, --num:    対局数の上限 (None の場合は全対局を対象)
"""

import argparse
import concurrent.futures
import random
import sys
import warnings

import numpy as np
import pandas as pd
import shogi
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

# 定数
RESULT_HEADER = (
    # 直前の局面を表す sfen
    "prev_sfen",
    # 現在の局面を表す sfen
    "sfen",
    # 正解確率
    "probability",
    # 自駒の配置を表す特徴量 (14, 9, 9)
    "loc",
    # 敵駒の配置を表す特徴量 (14, 9, 9)
    "opp_loc",
    # 持ち駒の特徴量(7)
    "hand",
    # 敵の持ち駒の特徴量(7)
    "opp_hand",
)

# パラメータ
USER_NAME = "garden0905"
INCORRECT_DATA_NUM = 2
MOVE_NUMBER_LIMIT = 100


def extract_features(board: shogi.Board, side: str) -> dict:
    """
    盤面の特徴量を抽出する関数
    Args:
        board (shogi.Board): 盤面情報
        side (str): 評価対象の色
    Returns:
        dict: 引数 board から抽出した特徴量
    """
    features = {
        "loc": np.zeros((14, 9, 9), dtype=np.int8),
        "opp_loc": np.zeros((14, 9, 9), dtype=np.int8),
        "hand": np.zeros(7, dtype=np.int8),
        "opp_hand": np.zeros(7, dtype=np.int8),
    }

    # 各マスにある駒を特徴量に変換
    for i in range(9):
        for j in range(9):
            if side == shogi.BLACK:
                piece = board.piece_at(i * 9 + j)
            else:
                piece = board.piece_at((8 - i) * 9 + (8 - j))
            if piece is not None:
                if piece.color == side:
                    features["loc"][piece.piece_type - 1, i, j] = 1
                else:
                    features["opp_loc"][piece.piece_type - 1, i, j] = 1
    # 持ち駒情報を追加
    if side == shogi.BLACK:
        hand = board.pieces_in_hand[shogi.BLACK]
        opp_hand = board.pieces_in_hand[shogi.WHITE]
    else:
        hand = board.pieces_in_hand[shogi.WHITE]
        opp_hand = board.pieces_in_hand[shogi.BLACK]

    for piece_type, count in hand.items():
        features["hand"][piece_type - 1] = count

    for piece_type, count in opp_hand.items():
        features["opp_hand"][piece_type - 1] = count

    features = {key: value.tolist() for key, value in features.items()}
    return features


def gen_correct_data(board: shogi.Board, prev_sfen: str, side: int) -> pd.DataFrame:
    """正解データを生成する関数"""
    row = {
        "prev_sfen": prev_sfen,
        "sfen": board.sfen(),
        "probability": 1.0,
    } | extract_features(board, side)
    return pd.DataFrame([row] * INCORRECT_DATA_NUM)


def gen_incorrect_data(board: shogi.Board, side: int) -> pd.DataFrame:
    """不正解データを生成する関数"""
    rows = []
    prev_sfen = board.sfen()

    # 合法手をランダムに選択
    legal_moves = list(board.legal_moves)
    moves = random.sample(legal_moves, min(INCORRECT_DATA_NUM, len(legal_moves)))
    for move in moves:
        board.push(move)
        rows.append(
            {
                "prev_sfen": prev_sfen,
                "sfen": board.sfen(),
                "probability": 0.0,
            }
            | extract_features(board, side)
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
        prev_sfen = board.sfen()
        board.push_usi(move)
        if board.is_game_over() or board.move_number > MOVE_NUMBER_LIMIT:
            break
        if side == board.turn:
            data = gen_incorrect_data(board, side)
        else:
            data = gen_correct_data(board, prev_sfen, side)
        df = pd.concat([df, data], axis=0)

    return df


def main(args):
    # データフレームの読み込み
    df = pd.read_csv(args.input)

    if args.num is not None:
        df = df.sample(n=args.num)

    # 特徴量の抽出の並列処理
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(
                    extract_sfen,
                    df["sfen"].values,
                    df["side"].map({"black": shogi.BLACK, "white": shogi.WHITE}).values,
                ),
                total=len(df),
            )
        )

    # 結果の結合
    print("Concatenating results...", file=sys.stderr)
    result_df = pd.concat(results, axis=0)
    print(f"length of result data: {len(result_df)}", file=sys.stderr)

    # データの保存
    result_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="data/merged.csv")
    parser.add_argument("-o", "--output", type=str, default="data/extracted.csv")
    parser.add_argument("-n", "--num", type=int, default=None)
    args = parser.parse_args()
    main(args)
