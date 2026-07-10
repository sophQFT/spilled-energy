# Spilled Energy 再現実装ノート

- 論文: Spilled Energy in Large Language Models (ICLR 2026, arXiv:2602.18671)
- 公式コード: https://github.com/OmnAI-Lab/spilled-energy
- 自分の fork: https://github.com/sophQFT/spilled-energy
- 環境: 研究室サーバー y60 / RTX 4090 (24GB) / CUDA 12.8
- 作業ブランチ: replication

## 2026-07-05: 環境構築

### 1. サーバーへ接続（接続先・ユーザー名は各自環境に置き換え）

```bash
ssh <username>@<server-address>
```

例：

```bash
ssh username@192.168.1.xx
```

---

## 2. リポジトリの場所

サーバー上では，リポジトリは以下に clone 済み。

```bash
~/paper-replication/spilled-energy
```

作業ディレクトリへ移動する。

```bash
cd ~/paper-replication/spilled-energy
```

---

## 3. 作業ブランチを確認

この再現実装では `replication` ブランチで作業する。

```bash
git branch
```

`replication` に切り替える場合：

```bash
git switch replication
```

---

## 4. ログイン後の定型作業

サーバーにログインしたら，まず以下を実行する。

```bash
cd ~/paper-replication/spilled-energy
source .venv/bin/activate
```

仮想環境が有効になると，プロンプトの先頭に `(.venv)` が表示される。

---

## 5. 作業前にGitの状態を確認

変更途中のファイルがないか確認する。

```bash
git status
```

GitHub側の最新状態を取り込む場合：

```bash
git pull
```

---

## 6. GPUを確認する場合

GPUを使う実験の前に確認する。

```bash
nvidia-smi
```

---

## 7. メモを編集する

このファイルを編集する場合：

```bash
nano notes/00_setup.md
```

nano の保存・終了：

```text
Ctrl + O   保存
Enter      保存確定
Ctrl + X   終了
```

---

## 8. 変更をGitHubに反映する

変更内容を確認する。

```bash
git status
```

変更したファイルをステージする。

```bash
git add notes/00_setup.md
```

コミットする。

```bash
git commit -m "Add setup notes"
```

GitHubに送る。

```bash
git push
```

---

## 9. 作業再開時の最小手順

久しぶりに作業を再開するときは，基本的にこれを実行する。

```bash
ssh <username>@<server-address>
cd ~/paper-replication/spilled-energy
source .venv/bin/activate
git status
```

必要に応じて：

```bash
git pull
nvidia-smi
```

---
