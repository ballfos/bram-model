"""
fetch.py

Description:
    対象のタグと段位の棋譜データを収集するスクリプト

Options:
    -o, --output: 出力ファイルのパス
    -r, --request-size: 1リクエストあたりの取得件数
"""

import argparse
import csv
import itertools

import requests
from tqdm import tqdm

# 定数
RESULT_HEADER = (
    "side",
    "sfen",
)

TARGET_TAGS = (
    "角道オープン四間飛車",
    "角交換振り飛車",
    "やばボーズ流",
)
TARGET_GRADES = (
    "一級",
    "初段",
    "二段",
    "三段",
)
RANGE_SIZE = 30000


def fetch_records(tag, grade, request_size):
    url = f"https://www.shogi-extend.com/api/lab/swars/cross-search.json?back_to=%2Fswars%2Fsearch&x_tags[]={tag}&x_tags_cond_key=or&x_grade_keys[]={grade}&x_judge_keys=__empty__&x_location_keys=__empty__&x_style_keys=__empty__&x_user_keys=__empty__&y_tags=__empty__&y_tags_cond_key=and&x_grade_diff_key=__empty__&y_grade_keys=__empty__&y_style_keys=__empty__&y_user_keys=__empty__&xmode_keys[]=%E9%87%8E%E8%89%AF&imode_keys[]=normal&rule_keys[]=ten_min&preset_keys[]=%E5%B9%B3%E6%89%8B&final_keys=__empty__&query=%E6%89%8B%E6%95%B0:%3E%3D80&range_size={RANGE_SIZE}&request_size={request_size}&open_action_key=new_tab&download_key=off&bg_request_key=off&bookmark_url_key=off&fetch_index=2"
    response = requests.get(url)
    data = response.json()
    query = data["redirect_to"]["to"].split("?")[1]
    url = f"https://www.shogi-extend.com/w.json?{query}&per=100"
    response = requests.get(url)
    data = response.json()
    return data["records"]


def get_side(users, tag):
    for user in users:
        if tag in user["attack_tag_list"]:
            return user["location_key"]
    return "black"


def main(args):
    # 出力ファイルの初期化
    with open(args.output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_HEADER)

    # タグ，段位毎にデータを取得
    for tag, grade in tqdm(
        itertools.product(TARGET_TAGS, TARGET_GRADES),
        total=len(TARGET_TAGS) * len(TARGET_GRADES),
    ):
        # 棋譜レコードの取得
        records = fetch_records(tag, grade, args.request_size)
        tqdm.write(f"num: {len(records):4d}, tag: {tag}, grade: {grade}")

        # 各レコードから局面データを抽出
        rows = []
        for record in records:
            rows.append(
                (
                    get_side(record["memberships"], tag),
                    record["sfen_body"],
                )
            )

        # データの保存
        with open(args.output, "a") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="data/fetched.csv")
    parser.add_argument("-r", "--request-size", type=int, default=50)
    args = parser.parse_args()
    main(args)
