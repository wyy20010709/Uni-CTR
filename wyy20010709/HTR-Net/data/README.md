# Data layout

Expected domain folders:
- `fashion/raw`, `fashion/split`
- `beauty/raw`, `beauty/split`
- `gift/raw`, `gift/split`

Raw input is Amazon 5-core JSON lines (`.json`/`.jsonl`).

After preprocessing, each split file is TSV with columns:
`domain user_id item_id label timestamp`
