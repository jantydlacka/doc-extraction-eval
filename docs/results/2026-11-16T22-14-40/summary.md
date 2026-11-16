# Run summary

- run: 2026-11-16T22-14-40
- tool commit: da2ee3954ec0aace762f0d544cca670c7a4fca55
- mode: real
- prompts: v1, v2
- cases: 20 (all)
- reps: 1
- prices verified: 2026-11-16 (data/prices.json)
- models: claude-haiku-5-5, claude-sonnet-5-5
- effort: medium
- max_tokens: 4096
- estimate: up to 2.0108 USD
- run limit: 2.02 USD
- ledger: ~/.local/share/llmeval/spend.jsonl
- rows (case × prompt × model × rep): 80; wall time: 384.8 s

## Pass rate

Counts are rows (case × rep). Cost is "-" where no call in the group was priced.

| model | prompt | passed rows | cost USD | cost per passed row | mean cost per row | median latency s |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | v1 | 20/20 (100%) | 0.013061 | 0.000653 | 0.000653 | 4.33 |
| claude-sonnet-5-5 | v1 | 20/20 (100%) | 0.211020 | 0.010551 | 0.010551 | 4.66 |
| claude-haiku-5-5 | v2 | 19/20 (95%) | 0.013920 | 0.000733 | 0.000696 | 4.51 |
| claude-sonnet-5-5 | v2 | 20/20 (100%) | 0.230920 | 0.011546 | 0.011546 | 4.53 |

## Tokens per row

Means over the rows whose call returned usage; thinking tokens are part of output.

| model | prompt | rows with usage | mean input tokens | mean output tokens |
|---|---|---|---|---|
| claude-haiku-5-5 | v1 | 20/20 | 2806.0 | 744.9 |
| claude-sonnet-5-5 | v1 | 20/20 | 2806.0 | 493.9 |
| claude-haiku-5-5 | v2 | 20/20 | 3298.0 | 732.4 |
| claude-sonnet-5-5 | v2 | 20/20 | 3298.0 | 495.0 |

## Pass rate per field

Rows (case × rep) with the field ok. A row whose call did not end normally (max_tokens, refusal, no response) or whose output is not a JSON object counts as failed on every field; runs.csv marks such rows with fields_counted=false.

| field | claude-haiku-5-5 v1 | claude-sonnet-5-5 v1 | claude-haiku-5-5 v2 | claude-sonnet-5-5 v2 |
|---|---|---|---|---|
| cislo_smlouvy | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| isin | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| isin_platny | 20/20 (100%) | 20/20 (100%) | 19/20 (95%) | 20/20 (100%) |
| castka | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| mena | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| citelne | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |

## Per case (passing repetitions)

| case | claude-haiku-5-5 v1 | claude-sonnet-5-5 v1 | claude-haiku-5-5 v2 | claude-sonnet-5-5 v2 |
|---|---|---|---|---|
| c01 | 1/1 | 1/1 | 1/1 | 1/1 |
| c02 | 1/1 | 1/1 | 1/1 | 1/1 |
| c03 | 1/1 | 1/1 | 0/1 | 1/1 |
| c04 | 1/1 | 1/1 | 1/1 | 1/1 |
| c05 | 1/1 | 1/1 | 1/1 | 1/1 |
| c06 | 1/1 | 1/1 | 1/1 | 1/1 |
| c07 | 1/1 | 1/1 | 1/1 | 1/1 |
| c08 | 1/1 | 1/1 | 1/1 | 1/1 |
| c09 | 1/1 | 1/1 | 1/1 | 1/1 |
| c10 | 1/1 | 1/1 | 1/1 | 1/1 |
| c11 | 1/1 | 1/1 | 1/1 | 1/1 |
| c12 | 1/1 | 1/1 | 1/1 | 1/1 |
| c13 | 1/1 | 1/1 | 1/1 | 1/1 |
| c14 | 1/1 | 1/1 | 1/1 | 1/1 |
| c15 | 1/1 | 1/1 | 1/1 | 1/1 |
| c16 | 1/1 | 1/1 | 1/1 | 1/1 |
| c17 | 1/1 | 1/1 | 1/1 | 1/1 |
| c18 | 1/1 | 1/1 | 1/1 | 1/1 |
| c19 | 1/1 | 1/1 | 1/1 | 1/1 |
| c20 | 1/1 | 1/1 | 1/1 | 1/1 |

## Baselines

| baseline | passed |
|---|---|
| oracle | 20/20 (100%) |
| all-null | 0/20 (0%) |

## Prompts against v1, paired per case

- claude-haiku-5-5, v2 vs v1: improved 0 (-), worse 1 (c03), unchanged 19
- claude-sonnet-5-5, v2 vs v1: improved 0 (-), worse 0 (-), unchanged 20
- Note: 20 cases × 1 repetition(s) and non-deterministic output; a difference of a case or two may be noise, not an effect of the prompt.

## Stopped at max_tokens

- none

## Warnings

response_model_differs: the response names another model than requested; the row is priced as the requested one. prompt_over_estimate: the billed prompt is above count_tokens with its margin, so the bound the budget checked was too low.

- none

## Failures

- c03 v2 claude-haiku-5-5 #1: isin_platny: spatne (False, expected True)

## Spend

- cost of the graded calls: 0.468920 USD
- written to the ledger by this run (cost_usd): 0.468920 USD
- possible unrecorded spend (attempts without a response, recorded at 0): up to 0.000000 USD; counted at this bound toward the cap and the run limit (2.02 USD)
- ledger now: 0.526180 USD counted toward the 5 USD cap (cost_usd sum 0.526180 USD; 0 line(s) with billed "unknown" counted at their bound_usd)
