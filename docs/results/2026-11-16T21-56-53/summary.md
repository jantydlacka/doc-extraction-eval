# Run summary

- run: 2026-11-16T21-56-53
- tool commit: da2ee3954ec0aace762f0d544cca670c7a4fca55
- mode: real
- prompts: v1
- cases: 1: c17
- reps: 1
- prices verified: 2026-11-16 (data/prices.json)
- models: claude-sonnet-5-5
- effort: medium
- max_tokens: 4096
- estimate: up to 0.0469 USD
- run limit: 1 USD
- ledger: ~/.local/share/llmeval/spend.jsonl
- rows (case × prompt × model × rep): 1; wall time: 24.3 s

## Pass rate

Counts are rows (case × rep). Cost is "-" where no call in the group was priced.

| model | prompt | passed rows | cost USD | cost per passed row | mean cost per row | median latency s |
|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | v1 | 0/1 (0%) | 0.046150 | - | 0.046150 | 24.02 |

## Tokens per row

Means over the rows whose call returned usage; thinking tokens are part of output.

| model | prompt | rows with usage | mean input tokens | mean output tokens |
|---|---|---|---|---|
| claude-sonnet-5-5 | v1 | 1/1 | 2595.0 | 4096.0 |

## Pass rate per field

Rows (case × rep) with the field ok. A row whose call did not end normally (max_tokens, refusal, no response) or whose output is not a JSON object counts as failed on every field; runs.csv marks such rows with fields_counted=false.

| field | claude-sonnet-5-5 v1 |
|---|---|
| cislo_smlouvy | 0/1 (0%) |
| isin | 0/1 (0%) |
| isin_platny | 0/1 (0%) |
| castka | 0/1 (0%) |
| mena | 0/1 (0%) |
| citelne | 0/1 (0%) |

## Per case (passing repetitions)

| case | claude-sonnet-5-5 v1 |
|---|---|
| c17 | 0/1 |

## Baselines

| baseline | passed |
|---|---|
| oracle | 1/1 (100%) |
| all-null | 0/1 (0%) |

## Stopped at max_tokens

- c17 v1 claude-sonnet-5-5 #1: output cut at max_tokens=4096

## Warnings

response_model_differs: the response names another model than requested; the row is priced as the requested one. prompt_over_estimate: the billed prompt is above count_tokens with its margin, so the bound the budget checked was too low.

- none

## Failures

- c17 v1 claude-sonnet-5-5 #1: output cut at max_tokens=4096; invalid JSON: Expecting value: line 1 column 1 (char 0)

## Spend

- cost of the graded calls: 0.046150 USD
- written to the ledger by this run (cost_usd): 0.046150 USD
- possible unrecorded spend (attempts without a response, recorded at 0): up to 0.000000 USD; counted at this bound toward the cap and the run limit (1 USD)
- ledger now: 0.046790 USD counted toward the 5 USD cap (cost_usd sum 0.046790 USD; 0 line(s) with billed "unknown" counted at their bound_usd)
