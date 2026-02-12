# HTR-Net (Uni-CTR tiny smoke framework)

该项目在 `Uni-CTR` 风格下提供可复现训练/评测骨架，支持：
- HTR-Net
- SharedBottom / MMOE / PLE / STAR baselines

> 当前阶段仅用于 tiny 数据冒烟，目标是跑通训练、验证、测试、checkpoint、AUC+Logloss。

## 1. 安装
```bash
cd wyy20010709/HTR-Net
pip install -r requirements.txt
```

## 2. 数据准备
- Uni-CTR 原始数据可放到 `data/raw/`（后续用户自行放全量）
- tiny 数据由脚本生成到 `data/tiny/`

### 2.1 生成 tiny 数据（无原始数据时用 toy 模式）
```bash
python scripts/make_tiny_from_unictr.py --unictr_root data/raw --output_dir data/tiny/amazon_book_movie --toy_if_missing
```

### 2.2 统计数据
```bash
python scripts/summarize_dataset.py --data_dir data/tiny/amazon_book_movie --as_json
```

## 3. 训练与评测
统一入口：
```bash
python -m htrnet.trainers.trainer --config configs/tiny_amazon_book_movie.yaml --model htr_net
python -m htrnet.trainers.trainer --config configs/tiny_amazon_book_movie.yaml --model sharedbottom
python -m htrnet.trainers.trainer --config configs/tiny_amazon_book_movie.yaml --model mmoe
python -m htrnet.trainers.trainer --config configs/tiny_amazon_book_movie.yaml --model ple
python -m htrnet.trainers.trainer --config configs/tiny_amazon_book_movie.yaml --model star
```

输出位于 `outputs/<exp>/`：
- `metrics.json`
- `log.txt`
- `best.pt`
- `last.pt`

## 4. 一键 smoke
```bash
bash scripts/run_tiny_smoke.sh
```

## 5. 多源方案A入口
配置：`configs/tiny_multisource_schemeA.yaml`
- 采用 super_source + target 的接口与占位参数（两阶段参数保留）。
- 本阶段不强制执行复杂多阶段全量实验。
