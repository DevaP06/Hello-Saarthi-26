# Archived dataset items

Items removed from the active evaluation set (`../dataset.jsonl`). Nothing in this folder is read by
`validate_dataset.py`, by the tests' active-dataset checks, or by any future runner.

## removed_hi_mr_2026-11-23.jsonl

`ml-01` (Hindi), `ml-03` (Marathi) and `ml-04` (Hindi), copied verbatim from the Sprint 1 dataset
(schema as of Sprint 1: `forbidden_claims` as plain strings, no `expected_answer_language`).

Removed in the Sprint 1 correction pass (2026-11-23): the active set now supports English and Hinglish
only. Open points recorded by the dataset review before removal, kept here for whoever restores them:

- All three need a native-speaker review (grammar, natural phrasing).
- ml-01: the key-point pattern missed common spellings ("दो गुना", "दुगना").
- ml-03: it dropped the "full-scale sine wave" condition of its twin df-07, its pattern missed
  "डेसिबल" / "decibels", and it used Devanagari digits (१६), which confounds language with numerals.
- ml-04: the regression case for audit finding H1 (the coverage gate treats Devanagari questions as
  covered). Its forbidden claim ("France won the 2018 World Cup") is true but unsupported, so if
  restored it must use `{"text": ..., "reason": "unsupported"}`.

To restore an item, migrate it to the current schema first; the validator will reject the old shape.
