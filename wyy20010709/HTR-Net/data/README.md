# Data layout

- `raw/`: place Uni-CTR raw/full files here (optional in smoke phase).
- `tiny/`: generated tiny sampled datasets for smoke tests.

Expected minimal columns in tiny splits:
- domain_id
- user_id
- item_id
- label
- text (optional)
