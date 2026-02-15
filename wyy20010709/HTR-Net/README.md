# HTR-Net (Uni-CTR style)

在 Uni-CTR 仓库根目录（包含 `wyy20010709/` 那层）执行。

## 1) 准备目录
```powershell
$base = "wyy20010709/HTR-Net/data"
New-Item -ItemType Directory -Force -Path `
  "$base/fashion/raw", "$base/fashion/split", `
  "$base/beauty/raw",  "$base/beauty/split",  `
  "$base/gift/raw",    "$base/gift/split" | Out-Null
```

## 2) 拷贝 Amazon 5-core JSON lines 到 raw
- `wyy20010709/HTR-Net/data/fashion/raw/AMAZON_FASHION_5.json`
- `wyy20010709/HTR-Net/data/beauty/raw/All_Beauty_5.json`
- `wyy20010709/HTR-Net/data/gift/raw/Gift_Cards_5.json`

## 3) 预处理 + 8:1:1 切分（JSONL）
```bash
python wyy20010709/HTR-Net/scripts/preprocess_and_split.py \
  --data_root wyy20010709/HTR-Net/data \
  --domains fashion beauty gift \
  --seed 2026 \
  --split_ratio 0.8 0.1 0.1 \
  --input_format jsonl \
  --rating_field overall \
  --user_field reviewerID \
  --item_field asin \
  --time_field unixReviewTime \
  --label_rule gt3
```

输出文件：
- `data/<domain>/split/train.tsv`
- `data/<domain>/split/val.tsv`
- `data/<domain>/split/test.tsv`

列格式固定：`domain user_id item_id label timestamp`

## 4) 指定 source/target/model 训练（不随机）
统一入口：`wyy20010709/HTR-Net/run.py`。

```bash
python wyy20010709/HTR-Net/run.py \
  --config wyy20010709/HTR-Net/configs/fast.yaml \
  --model ple \
  --source_domain beauty \
  --target_domain gift
```

互换源目标：
```bash
python wyy20010709/HTR-Net/run.py --model ple --source_domain gift --target_domain beauty
```

切换模型：
```bash
python wyy20010709/HTR-Net/run.py --model star --source_domain beauty --target_domain gift
python wyy20010709/HTR-Net/run.py --model mmoe --source_domain beauty --target_domain gift
python wyy20010709/HTR-Net/run.py --model sharedbottom --source_domain beauty --target_domain gift
python wyy20010709/HTR-Net/run.py --model htr_net --source_domain beauty --target_domain gift
```

## 5) 输出
每个 epoch 记录并输出：
- source AUC / Logloss
- target AUC / Logloss

保存：
- `wyy20010709/HTR-Net/outputs/<exp>/log.txt`
- `wyy20010709/HTR-Net/outputs/<exp>/metrics.json`
- `wyy20010709/HTR-Net/outputs/<exp>/ckpt_best.pt`
- `wyy20010709/HTR-Net/outputs/<exp>/ckpt_last.pt`
