# ollama-rubric

I built this because `ollama-bench -decision` in Ollama's `cmd/bench` grades fixed multiple-choice answers, whereas most fine-tunes I actually care about generate free-form text. Here, the model has to fix a learner's Spanish sentence if there is an error, explain why in Korean, and leave already-correct sentences alone.

Zero dependencies outside Python's standard library. It talks to local Ollama over `/api/generate` at temperature 0.

The 208-item eval set and fine-tuned models come from my tutor repo: <https://github.com/Riloox/gemma-spanish-tutor-wip>.

## Overview

`rubric` runs a list of local Ollama models against a frozen JSONL task file and outputs:

1. Pass counts with Wilson 95% intervals and p50 latency per model.
2. Deterministic string checks (`contains_any`, `contains_all`, `not_contains`, `says_correct`, `require_korean`).
3. An optional local LLM judge (`--judge gemma4:26b`) using structured JSON output (`format` schema), blind to the reference fix by default (`--judge-show-fix` opts in).
4. Per-item failure diffs and judge-vs-string-check disagreement rows.
5. Stratified blind sampling and Cohen's kappa (`calibrate --interactive`) so you can verify the judge against 30 human labels before trusting a score table.

## Measured results (208 items)

Task file: `tasks/test_set_v4.1.jsonl` (104 erroneous Spanish sentences expecting a targeted fix, 104 already-correct sentences expecting confirmation without invented edits, all requiring Korean explanations). Everything below ran on Ollama `0.40.7` with `think=false` and `temperature=0.0`.

### Prompt format vs fine-tuning

With the plain tutor prompt, `gemma4-tutor-v4` beat stock `gemma4:e2b` by 62 points (168 vs 106). Switching to `prompts/tutor_system_rewrite_first.txt` moved the stock model to 170/208 and showed that `v4` had regressed on erroneous sentences (75 vs 84). That led to the `v5` self-distillation run using `gemma4:26b` fallback on hard error items, which recovered errors to 90/104.

| Model | System prompt | Total | Errors (104) | Correct (104) | p50 |
| --- | --- | --- | --- | --- | --- |
| `gemma4:e2b` (`b37049369adf`) | plain | 106/208 | 84 | 22 | 2.01s |
| `gemma4-tutor-v4` (`7da4a1bb1fa7`) | plain | 168/208 | 84 | 84 | 1.00s |
| `gemma4:e2b` (`b37049369adf`) | rewrite-first | 170/208 | 84 | 86 | 0.72s |
| `gemma4-tutor-v4` (`7da4a1bb1fa7`) | rewrite-first | 174/208 | 75 | 99 | 0.50s |
| `gemma4-tutor-v5` (`b625e3338088`) | rewrite-first | 184/208 | 90 | 94 | 0.99s |

### Local judge (gemma4:26b)

Re-scoring those rewrite-first answers with `--judge gemma4:26b --judge-think false` (digest `001e5dafc3c7`, blind to reference fixes, 0 parse errors across 312 calls):

| Model | Pass | Rate | 95% CI | Errors (104) | Correct (104) | p50 | Disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma4:e2b` | 158/208 | 76.0% | 69.7%-81.3% | 84/104 | 74/104 | 0.72s | 15 |
| `gemma4-tutor-v4` | 166/208 | 79.8% | 73.8%-84.7% | 75/104 | 91/104 | 0.50s | 10 |
| `gemma4-tutor-v5` | 181/208 | 87.0% | 81.8%-90.9% | 90/104 | 91/104 | 0.99s | 6 |

### Disagreement examples

Running both string checks and the local judge side-by-side catches errors neither sees alone:

- `gemma4-tutor-v4` on `t4-ok-001` (`mech_pass_judge_fail`): the model opened with `**바뀐 곳:** 없음` and `네, 맞아요!` (passing `says_correct`), then told the learner to replace `visitar` with `ir` in the explanation. The judge failed it.
- `gemma4-tutor-v5` on `t4-ok-040` (`mech_pass_judge_fail`): confirmed `"A mis amigos les gustaban los perros de la calle."` was grammatically right, then translated `perros` as `길고양이들` (street cats). The judge caught the mistranslation.
- `gemma4-tutor-v5` on `t4-ok-054` (`mech_pass_judge_fail`): confirmed `"Te doy este mapa para que no te pierdas en el camping."` was right, then claimed `pierdas` is an infinitive (`원형`). The judge caught the wrong grammar term.
- `gemma4-tutor-v5` on `t4-ok-018` (`mech_fail_judge_pass`): answered `네, 어색하지 않아요!` ("Yes, it's not awkward!"). `says_correct` failed it because `어색` sits in the denial word list, and the judge passed it.

## Quick start

```bash
# 1. Inspect a task file
python3 rubric.py list --tasks tasks/test_set_v4.1.jsonl

# 2. Score models against local Ollama (needs `ollama serve`)
python3 rubric.py run \
  --tasks tasks/test_set_v4.1.jsonl \
  --models gemma4:e2b,gemma4-tutor-v4,gemma4-tutor-v5 \
  --system prompts/tutor_system_rewrite_first.txt \
  --think false \
  --judge gemma4:26b \
  --out results.json

# 3. Re-grade saved answers without re-running generation
python3 rubric.py rescore \
  --results results.json \
  --tasks tasks/test_set_v4.1.jsonl \
  --judge gemma4:26b \
  --out results-judged.json

# 4. Blind 30-item human calibration (computes Cohen's kappa)
python3 rubric.py calibrate \
  --results results-judged.json \
  --interactive \
  --out calibration.jsonl
```

Try the full pipeline with synthetic answers and no Ollama server:

```bash
python3 rubric.py run \
  --tasks tasks/test_set_v4.1.jsonl \
  --models fake-a,fake-b \
  --fake fake-a \
  --judge fake-judge \
  --limit 40
```

Run the unit suite (`python3 tests.py`).

## Limits

- Judge calibration is pending: I do not speak Korean, so a Korean speaker still needs to run the 30-item blind pass (`calibrate --interactive`, 15 disagreements, 7 both-pass, 8 both-fail from `results-v5-judged.json`) to get human-vs-judge Cohen's kappa.
- Blind judge vs system prompt friction: on 12 `gemma4:e2b` items, the model used the prompt's `**다시 써 보면:**` header without emitting `**바뀐 곳:** 없음`, and `gemma4:26b` treated the heading itself as a rewrite. Showing mechanical checks and judge verdicts side-by-side makes that visible.
- Standalone Python over `/api/generate`: this sits outside `cmd/bench` so any JSONL task file runs without rebuilding Go. Adding `-rubric` inside `cmd/bench` would make sense if maintainers want it upstream.

## Layout

```
rubric.py        CLI: run, rescore, list, calibrate, comparison table, Wilson CI
scoring.py       deterministic checks, structured Ollama judge, disagreement + Cohen's kappa
ollama_http.py   stdlib urllib client for /api/generate (with JSON schema format) + FakeForTasks
tests.py         plain asserts (`python3 tests.py`)
tasks/           frozen 208-item Korean->Spanish eval set (test_set_v4.1.jsonl)
prompts/         rewrite-first tutor system prompt
```
