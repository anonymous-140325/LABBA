# LLM-Assisted Blow-Pattern Biometric Authentication

`llmEvaluation.py` is a research prototype for authenticating users from "blow"
audio recordings (short breath/blow bursts captured as an audio biometric).
It combines a classical embedding/cosine-similarity matcher with a local LLM
(via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)) that
is only consulted for borderline decisions.

## How it works

1. **Load embeddings** — one audio embedding vector per recording session
   (see [Data format](#data-format) below), used to build a cosine
   similarity matrix across all sessions.
2. **Load structured features** — per-blow and per-session features
   (timing, amplitude, rise/fall shape) extracted separately from the same
   recordings.
3. **Per-user adaptive threshold** — for each enrolled user, a
   cosine-similarity threshold is derived from that user's own session-to-session
   variability (`determine_threshold`, `get_all_user_thresholds`).
4. **Score enroll vs. attempt** — for each pair, one of two interchangeable
   scoring methods converts the enroll/attempt features into what the LLM
   sees, selected via the `SCORING_METHOD` constant near `create_auth_prompt()`:
   - `"score_based"` (default) — continuous 0-100 similarity scores,
     each normalized against that user's own per-session baseline
     (z-score for scalars, DTW-distance-vs-baseline for sequences).
   - `"rule_based"` — discrete `VERY_GOOD`/`GOOD`/`OKAY`/`WEAK`/`BAD`
     labels from fixed, global percentage-difference cutoffs — no
     per-user baseline involved.

   Both read the same `fused_features()` output and produce the same
   `Decision`/`Confidence`/`Reason` prompt contract, so switching between
   them only requires changing `SCORING_METHOD` — nothing else in the
   pipeline changes. `SCORING_METHOD` is folded into the checkpoint's run
   signature, so a checkpoint from one scoring method is never silently
   resumed under the other.
5. **Evaluation** — the script sweeps every session pair, tallies
   TP/FP/TN/FN, prints precision/recall/FPR/FNR/accuracy, and writes a
   per-pair breakdown to `results.csv`.

## Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+. `llama-cpp-python` compiles a native extension on
install; if you have a GPU and want offloaded inference, install it with the
appropriate build flags for your platform (see the
[llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python#installation)).

## Model

The script expects a GGUF-quantized Phi-3-mini model at:

```
assets/Phi-3-mini-4k-instruct-q4.gguf
```

(relative to `llmEvaluation.py` itself, alongside the `assets/LABBA_Data`
dataset described below — not under the OS home directory, so the path is
the same on every machine that checks out this repo). Download it (e.g.
`Phi-3-mini-4k-instruct-q4.gguf` from the
[microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
repo on Hugging Face) and place it at that path, or edit `model_path` in
`llmEvaluation.py`'s `__main__` block to point elsewhere. Any other instruct
GGUF model can be substituted — the chat template in `chat_prompt` uses
Phi-3's `<|system|>/<|user|>/<|assistant|>` tags, so adjust that if you swap
models.

## Data format

This repo does **not** include any biometric recordings or derived data —
only the code. To reproduce results you need a dataset laid out like
`assets/LABBA_Data/` (relative to `llmEvaluation.py`): one subfolder per
recording session, each holding a matched triple of files that share a
common `<User>_<Posture>_Session<N>` id, e.g. for `User10_Sit_Session1`:

- `User10_Sit_blow_features_Session1.json` — a list of per-blow objects:
  ```json
  {
    "blow_id": 1,
    "start_time": 0.55,
    "end_time": 2.70,
    "duration": 2.15,
    "mean_amp": 0.145,
    "rms_amp": 0.195,
    "std_amp": 0.054,
    "cv_amp": 0.374,
    "rise_time": 0.339,
    "fall_time": 1.815,
    "attack_slope": 0.851,
    "decay_slope": -0.156
  }
  ```
- `User10_Sit_session_features_Session1.json` — one object per session:
  ```json
  {
    "num_blows": 2,
    "start_delay": 0.55,
    "blow_gaps": [0.87],
    "session_duration": 5.46,
    "total_blow_duration": 4.04,
    "duty_cycle": 0.739
  }
  ```
- `User10_Sit_emb_Session1.csv` — the session's embedding vector, one float
  per line (e.g. 128-d):
  ```
  -0.2185
  0.0981
  ...
  ```

`read_all_features()`/`read_all_embeddings()` walk this directory
recursively (`os.walk`), pair files up by stripping each file's
`_blow_features` / `_session_features` / `_emb` marker to recover the
shared session id, and sort both resulting lists by that id using a
natural (numeric-aware) sort — a plain lexicographic sort would put
`User10_...` right after `User1_...`, ahead of `User2_...`. `__main__`
then checks `all_features` and `all_embeddings` are the same length and
aligned id-for-id before running anything else, since `session_names`
(built from `num_users`/`sessions_per_user`) labels sessions purely by
position: 500 sessions across 50 users, 10 sessions each (5 `Sit` + 5
`Stand`), matching `num_users = 50` / `sessions_per_user = 10` in
`__main__`.

## Raw audio (not included)

`assets/LABBA_Data/` contains only the derived, non-reversible artifacts
(per-blow/session features and embedding vectors) that `llmEvaluation.py`
consumes — no raw audio. `softBiometricExtraction.py` is the earlier stage
that produces those artifacts from raw "blow" audio recordings (see
`process_all_data()`/`analyze_blow_pattern()`), but the raw recordings
themselves are **not provided in this repo**, since they're biometric data
tied to real participants. Reproducing `LABBA_Data` from scratch would
require your own raw-audio dataset run through `softBiometricExtraction.py`.

## Running

```bash
python llmEvaluation.py
```

Key knobs:

| Variable | Meaning |
|---|---|
| `SCORING_METHOD` (module-level, just above `create_auth_prompt()`) | `"score_based"` (default) or `"rule_based"` — see [How it works](#how-it-works) above. |
| `num_users`, `sessions_per_user` (in `__main__`) | Used to synthesize `session_names` (`user1`, `user1`, ..., `user2`, ...) — must match your dataset's layout and sort order. |
| `k`, `q` (in `__main__`) | Neighbor/percentile parameters for the per-user threshold (`determine_threshold`). |
| `range(0, 500)` (x2) in the comparison loop | Number of sessions compared; set to `len(all_embeddings)` for your dataset. |
| `n_ctx`, `max_tokens`, `temperature` | Llama.cpp generation settings. |

Output: console metrics (precision/recall/false-positive/false-negative/
accuracy, plus average per-comparison timing) and `results.csv` with one row
per session pair (`decision`, `confidence`, `reason`, correctness flag).


## Notes

- This is research/prototype code for evaluating LLM-assisted biometric
  decisions, not a production authentication system.
