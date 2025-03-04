# bram-model

## 手順

### 1. 環境変数ファイルの作成

```bash
cp .env.sample .env
```

`BASE_IMAGE` を適切なイメージに設定

### 2. コンテナの起動

```bash
docker compose up -d --build
```

`--build` は初回のみ

### 3. devcontainer の起動

コマンドパレット (F1 or ⌘+⇧+P) から `Remote-Containers: Reopen in Container` を選択


## Note

| スクリプト           | 説明         |
| -------------------- | ------------ |
| `scripts/extract.py` | データの抽出 |
| `scripts/train.py`   | モデルの学習 |