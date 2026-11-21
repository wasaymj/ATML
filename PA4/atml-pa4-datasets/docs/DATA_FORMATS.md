# Dataset schemas and intended use

All saved JSONL records are one JSON object per line; `.jsonl.gz` is gzip-compressed JSONL. The split and source metadata must be preserved. **The files are not solution code.**

| Task | File / source | Key fields | Purpose |
|---|---|---|---|
| 1 | `public_locked/c4_calibration_locked.jsonl.gz` (pending) | `source_row`, `token_ids` | 128 independent 512-token C4 training sequences, used for Wanda/AWQ/sensitivity |
| 1 | `public_locked/c4_development_locked.jsonl.gz` (pending) | `source_row`, `token_ids` | 32 disjoint C4 training sequences for method selection |
| 1 | `public_locked/wikitext2_test_windows_locked.jsonl.gz` (pending) | `token_ids`, `score_target_offsets`, `scored_tokens` | 50,000 exact scored targets |
| 1 | `public_locked/c4_validation_windows_locked.jsonl.gz` (pending) | same | 50,000 exact scored targets |
| 1 | `public_locked/hellaswag_val_locked.jsonl.gz` (pending) | `order`, `example`, `source_row_index` | 2,000 HellaSwag validation examples; **first 512 are diagnostic subset** |
| 2B | `mqar/mqar_test.jsonl.gz` | `tokens`, `K`, `query_positions`, `answer_ids`, `source_positions`, `distances`, `regime` | 9,216 immutable, held-out test sequences |
| 2B | `mqar/mqar_development.jsonl.gz` | same | 2,304 development examples (256 per K × regime) |
| 2B | `scripts/mqar_generator.py` | `make_example(split, k, regime, index)` | deterministic training-stream generation |
| 2C | `kv_retrieval/kv_retrieval_cases.jsonl.gz` | `id`, `user_text`, `code`, `position`, `distractors`, `target_tokens`, `evidence_char_start`, `evidence_char_end` | 180 long-context retrieval prompts, gold code and evidence **character** offsets |
| 3 | `public_locked/gsm8k_train_locked.jsonl.gz` (pending) | `order`, `example`, `source_row_index` | 500 fixed training prompts; no GSM8K test examples |
| 3 | `public_locked/gsm8k_test_locked.jsonl.gz` (pending) | same | 60 held-out test questions |
| 3 | `short_form/short_train.jsonl` | `id`, `input`, `answer`, `family`, `split` | 300 training instructions |
| 3 | `short_form/short_test.jsonl` | same | 60 held-out instructions, 30 familiar and 30 shifted templates |
| 3 | `prefix_banks/prefix_banks_locked.jsonl.gz` (pending) | `prompt_id`, `bank`, `position`, `prompt`, `response_prefix_token_ids` | frozen target/original-draft/exact-speculative evaluation prefix banks |

## MQAR scoring

The correct value is labeled **at the query position**; the task is query-position classification, not next-token language modeling. Each test sequence contains K supervised query positions. The generator is deterministic for a given `(split, K, regime, index)`; do not regenerate the frozen `test` records. The reported character/token sequence and metadata are already aligned.

## KV spans and lengths

`evidence_char_start` / `evidence_char_end` refer to Python Unicode string slices into `user_text`, **not Qwen token indices**. The `target_tokens` field is a nominal target (2,048 or 4,096), not the measured chat-template token length. Staff must audit encoded lengths and full-cache correctness before release. Students should record actual token lengths after applying the pinned tokenizer and chat template.

## Public selections

Row indices in `public_manifests/` are **not sufficient** to freeze dataset contents. Immutable source commit IDs and row hashes live in `release_provenance.json` and materialized `_locked` records after staff publishing. Do not use an unpinned revision or silently substitute another split. The locked example data are authoritative; use the numeric manifests for audit and traceability.

## Prefix bank definitions

`teacher` and `exact_spec_original_gamma4` sample the *same target-prefix law* and are consistency controls; `original_draft` samples a different draft-prefix law. Each trained student's own policy-prefix records are **not** part of the release, because students must generate those during their experiments. Compare the same frozen prefixes across drafts. Response position is stored explicitly and must not be inferred from the number of saved records.
