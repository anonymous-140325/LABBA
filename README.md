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
4. **Evaluation** — the script sweeps every session pair, tallies
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
~/assets/Phi-3-mini-4k-instruct-q4.gguf
```

Download it (e.g. `Phi-3-mini-4k-instruct-q4.gguf` from the
[microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
repo on Hugging Face) and place it at that path, or edit `model_path` in
`llmEvaluation.py`'s `__main__` block to point elsewhere. Any other instruct
GGUF model can be substituted — the chat template in `chat_prompt` uses
Phi-3's `<|system|>/<|user|>/<|assistant|>` tags, so adjust that if you swap
models.

## Data format

This repo does **not** include any biometric recordings or derived data —
only the code. To reproduce results you need your own dataset laid out as
follows.

**Structured features** — under `~/assets/BlowPrintData/`, one subfolder per
session containing a matched pair of JSON files:

- `<base_name>_blow_features_fast.json` — a list of per-blow objects:
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
- `<base_name>_session_features_fast.json` — one object per session:
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

**Embeddings** — under `~/assets/BlowPrintData_clean/`, one CSV per session
whose filename contains `new_audio` (and not `binary`), with one float per
line (a flat embedding vector, e.g. 128-d):
```
-0.2185
0.0981
...
```

Both directories are walked recursively (`os.walk`), and each directory's
files are sorted alphabetically by filename to build `all_features` and
`all_embeddings` respectively — **the two lists must end up in the same
session order** for `session_names` (built from `num_users` /
`sessions_per_user` in `__main__`) to label the right session correctly.
If your embedding and feature filenames don't sort into matching order,
fix the sort key or the filenames before trusting the metrics — this
alignment is not otherwise checked at runtime.

## Running

```bash
python llmEvaluation.py
```

Key knobs at the top of `__main__`:

| Variable | Meaning |
|---|---|
| `num_users`, `sessions_per_user` | Used to synthesize `session_names` (`user1`, `user1`, ..., `user2`, ...) — must match your dataset's layout and sort order. |
| `k`, `q` | Neighbor/percentile parameters for the per-user threshold (`determine_threshold`). |
| `range(0, 500)` (x2) in the comparison loop | Number of sessions compared; set to `len(all_embeddings)` for your dataset. |
| `n_ctx`, `max_tokens`, `temperature` | Llama.cpp generation settings. |

Output: console metrics (precision/recall/false-positive/false-negative/
accuracy, plus average per-comparison timing) and `results.csv` with one row
per session pair (`decision`, `confidence`, `reason`, correctness flag).

Runtime is dominated by the LLM calls on `BORDERLINE` pairs only — most
pairs are resolved by the cosine threshold alone and never touch the model.

## Notes

- This is research/prototype code for evaluating LLM-assisted biometric
  decisions, not a production authentication system.
