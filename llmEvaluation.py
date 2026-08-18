import csv
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity


BORDERLINE_MARGIN = 0.3


# Dataset lives in an `assets/` folder alongside this script (not under the
# OS home directory) so the path is the same on every machine that checks
# out this repo, regardless of whose home directory it runs from.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LABBA_DATA_DIR = ASSETS_DIR / "LABBA_Data"

FEATURE_FILE_SUFFIX = ""

first_tp = True
first_tn = True
first_fp = True
first_fn = True


def natural_key(s):
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', s)]


def determine_threshold(matrix, k, p):
    """
    matrix: 2D numpy array
    k: the k-th nearest neighbor
    p: the p-th largest threshold candidate
    """
    threshold_candidates = []
    for i in range(matrix.shape[0]):
        distances = matrix[i, :]
        sorted_distances = np.sort(distances)
        threshold_candidates.append(sorted_distances[k])
    threshold_candidates = np.array(threshold_candidates)
    final_threshold = np.sort(threshold_candidates)[::-1][p]
    return final_threshold


def extract_submatrix(matrix, rownames, colnames, session_name):
    """
    matrix: 2D numpy array
    rownames: list of row names
    colnames: list of col names
    """
    row_indices = [i for i, r in enumerate(rownames) if r == session_name]
    col_indices = [j for j, c in enumerate(colnames) if c == session_name]

    if row_indices and col_indices:
        return matrix[np.ix_(row_indices, col_indices)]
    else:
        return None  # session name not found


def get_all_user_thresholds(matrix, session_names, k, p):
    unique_session_names = sorted(set(session_names), key=natural_key)
    user_matrices = {}

    for session in unique_session_names:
        subm = extract_submatrix(matrix, session_names, session_names, session)
        if subm is not None:
            user_matrices[session] = subm

    user_thresholds = {}
    for name, mat in user_matrices.items():
        user_thresholds[name] = determine_threshold(mat, k, p)

    return user_thresholds


def compute_cosine_similarity_matrix(all_embeddings):
    """
    Compute cosine similarity matrix from embeddings
    Returns similarity matrix (higher = more similar)
    """
    embedding_matrix = np.array([item['embedding'] for item in all_embeddings])

    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    similarity_matrix = cosine_similarity(embedding_matrix)

    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    return similarity_matrix


def cosine_similarity_to_distance(similarity_matrix):
    """
    Convert cosine similarity to distance
    Distance = 1 - similarity (range: 0 to 2, lower = more similar)
    """
    return 1 - similarity_matrix


# Every session in LABBA_Data lives under its own folder as a matched
# triple of files sharing one canonical id, e.g. for "User10_Sit_Session1":
#   User10_Sit_blow_features_Session1.json
#   User10_Sit_session_features_Session1.json
#   User10_Sit_emb_Session1.csv
# The marker below is the only thing distinguishing each file's name from
# that shared id, so stripping it recovers the id - which is how blow/
# session feature files get paired up, and how all_features/all_embeddings
# are checked for alignment in __main__.
BLOW_FEATURES_MARKER = "_blow_features"
SESSION_FEATURES_MARKER = "_session_features"
EMBEDDING_MARKER = "_emb"


def _canonical_session_id(stem):
    for marker in (
        f"{BLOW_FEATURES_MARKER}{FEATURE_FILE_SUFFIX}",
        f"{SESSION_FEATURES_MARKER}{FEATURE_FILE_SUFFIX}",
        EMBEDDING_MARKER,
    ):
        if marker in stem:
            return stem.replace(marker, "", 1)
    return stem


def read_all_embeddings():
    """
    Read all *_emb_*.csv embedding files from the LABBA_Data directory.
    """
    base_dir = LABBA_DATA_DIR

    embedding_files = []

    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if EMBEDDING_MARKER in filename and filename.lower().endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                embedding_files.append(file_path)

    # Natural (numeric-aware) sort so user2 sorts before user10 - a plain
    # lexicographic sort would put "user10_..." right after "user1_...",
    # which would misalign every session against session_names below.
    embedding_files.sort(key=lambda x: natural_key(os.path.basename(x).lower()))

    print(f"Found {len(embedding_files)} embedding files")
    return embedding_files


def load_all_embeddings_unformatted():
    """
    Load all embedding CSV files
    Each line in CSV is one data point
    """
    embedding_files = read_all_embeddings()

    all_embeddings = []

    for file_path in embedding_files:
        try:
            embedding = np.loadtxt(file_path, delimiter=',')

            filename = os.path.basename(file_path)
            stem = os.path.splitext(filename)[0]

            all_embeddings.append({
                'filename': _canonical_session_id(stem),
                'full_path': file_path,
                'embedding': embedding,
                'shape': embedding.shape,
                'dtype': embedding.dtype
            })

        except Exception as e:
            print(f"✗ Failed to load {file_path}: {e}")

    print(f"\n✓ Total embeddings loaded: {len(all_embeddings)}")

    return all_embeddings


def read_all_features():
    """
    Read all blow and session feature JSON files from the LABBA_Data
    directory, pairing each *_blow_features_*.json with its matching
    *_session_features_*.json by canonical session id.
    """
    base_dir = LABBA_DATA_DIR

    feature_files = []

    blow_marker = f"{BLOW_FEATURES_MARKER}{FEATURE_FILE_SUFFIX}"
    session_marker = f"{SESSION_FEATURES_MARKER}{FEATURE_FILE_SUFFIX}"

    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if blow_marker in filename and filename.endswith(".json"):
                blow_file = os.path.join(dirpath, filename)

                base_name = filename.replace(blow_marker, "", 1)
                session_filename = filename.replace(blow_marker, session_marker, 1)
                session_file = os.path.join(dirpath, session_filename)

                if os.path.exists(session_file):
                    feature_files.append({
                        'base_name': os.path.splitext(base_name)[0],
                        'blow_file': blow_file,
                        'session_file': session_file
                    })

    # Same natural-sort reasoning as read_all_embeddings() above.
    feature_files.sort(key=lambda x: natural_key(x['base_name'].lower()))

    print(f"Found {len(feature_files)} feature file pairs")
    return feature_files


def load_all_features_unformatted():
    """
    Load all feature files as raw JSON data (unformatted)
    Returns a list of dictionaries containing blow and session features
    """
    feature_files = read_all_features()

    all_data = []

    for item in feature_files:
        with open(item['blow_file'], 'r') as f:
            blow_features = json.load(f)

        with open(item['session_file'], 'r') as f:
            session_features = json.load(f)

        all_data.append({
            'filename': item['base_name'],
            'blow_features': blow_features,
            'session_features': session_features,
            'file_paths': {
                'blow_file': item['blow_file'],
                'session_file': item['session_file']
            }
        })

    print(f"✓ Loaded {len(all_data)} recordings (unformatted)")

    return all_data


def r(x, d=3):
    return round(float(x), d)


def fused_features(blow_features, session_features):
    """
    Per-session biometric representation feeding both scoring methods below
    (SCORING_METHOD == "score_based" or "rule_based" — see that constant
    near __main__):
    - rhythm sequence (gaps) -> gap DTW / gap sequence_similarity
    - duration sequence -> duration DTW / duration sequence_similarity
    - shape sequence (rise/fall ratio) -> shape DTW / shape_similarity
    - amplitude (rms) -> amplitude score (score-based only)
    - amplitude (mean) -> amplitude_similarity (rule-based only)
    - duty cycle -> "mass" score / mass_similarity
    - blow count -> blow_count score / blow_count_similarity
    - attack/decay slope sequences -> rule-based only, not currently
      compared by either scoring method's top-level score/label, kept here
      for parity with the rule-based lineage this was folded in from
    (the embedding/cosine score is computed separately from the embedding
    matrix, not part of this per-session structure — see
    cosine_position_score() / cosine_rule_based_label() below)
    """

    starts = np.array([b["start_time"] for b in blow_features])
    ends = np.array([b["end_time"] for b in blow_features])

    durations = [r(b["duration"]) for b in blow_features]
    mean_amp_seq = [r(b["mean_amp"]) for b in blow_features]
    rms_amp_seq = [r(b["rms_amp"]) for b in blow_features]

    rise_fall_seq = [
        r(b["rise_time"] / (b["fall_time"] + 1e-6))
        for b in blow_features
    ]
    attack_seq = [r(b["attack_slope"]) for b in blow_features]
    decay_seq = [r(b["decay_slope"]) for b in blow_features]

    gaps = []

    if len(starts) > 0:
        gaps.append(starts[0])
        for i in range(len(starts) - 1):
            gaps.append(starts[i + 1] - ends[i])

    gaps = np.array(gaps) if len(gaps) > 0 else np.array([0.0])
    gap_pattern = [r(x) for x in gaps]

    total_blow_duration = np.sum(durations)
    session_duration = r(session_features["session_duration"])
    num_blows = session_features["num_blows"]

    duty_cycle = total_blow_duration / (session_duration + 1e-6)

    duration_mean = r(np.mean(durations))
    duration_std = r(np.std(durations))
    amp_rms_mean = r(np.mean(rms_amp_seq))
    amp_mean = r(np.mean(mean_amp_seq))
    amp_std = r(np.std(mean_amp_seq))

    return {
        "num_blows": num_blows,
        "session_duration": session_duration,
        "total_blow_duration": r(total_blow_duration),
        "duty_cycle": r(duty_cycle),

        "gap_pattern": gap_pattern,

        "mean_amp_seq": mean_amp_seq,
        "amp_mean": amp_mean,
        "amp_std": amp_std,

        "rms_amp_seq": rms_amp_seq,
        "amp_rms_mean": amp_rms_mean,

        "duration_seq": durations,
        "duration_mean": duration_mean,
        "duration_std": duration_std,

        "rise_fall_seq": rise_fall_seq,
        "attack_seq": attack_seq,
        "decay_seq": decay_seq,
    }


# =========================================================
# Dynamic Time Warping — sequence-aware comparison
# =========================================================

def dtw_distance(seq1, seq2):
    """
    Elastic alignment distance between two numeric sequences of possibly
    different lengths. Unlike a mean (which discards order entirely) or a
    positional index-by-index diff (which breaks when lengths differ),
    DTW finds the cheapest way to align the two sequences end-to-end,
    letting one point match several on the other side. That's what makes
    it tolerant of a single blow that got segmented into two pieces in one
    session but not the other, while still being sensitive to genuine
    differences in rhythm/shape.

    Returns the total alignment cost normalized by path length, so
    distances are comparable across sequence-length combinations. Plain
    O(n*m) DP — sequences here are a handful of blows, so no windowing is
    needed for performance.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        # Exactly one side has no detected blows at all (e.g. a failed
        # recording) - there's no real pattern there to align against, so
        # this is maximally dissimilar, not a perfect (empty) match.
        return float("inf")
    if n == 1 and m == 1:
        return abs(float(seq1[0]) - float(seq2[0]))

    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        si = float(seq1[i - 1])
        for j in range(1, m + 1):
            cost = abs(si - float(seq2[j - 1]))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    path_len = n + m
    return float(D[n, m] / path_len)


def dtw_relative_z(seq1, seq2, baseline_dtw_scale):
    """
    DTW analogue of relative_z(): expresses the alignment distance between
    two sequences as a multiple of this user's own typical session-to-
    session DTW distance for that feature, instead of an absolute number.
    """
    return dtw_distance(seq1, seq2) / baseline_dtw_scale


# =========================================================
# Per-user baselines (replaces fixed global percent-diff cutoffs)
# =========================================================

# Scalar features: baseline = std of that value across a user's own
# sessions (same approach as v2).
SCALAR_BASELINE_KEYS = ["num_blows", "amp_rms_mean", "duty_cycle"]

# Sequence features: baseline = mean DTW distance between all pairs of a
# user's own sessions — "how much does this person's own rhythm/shape
# normally vary session to session," measured the same way (DTW) that a
# new attempt will be judged.
DTW_BASELINE_KEYS = ["gap_pattern", "duration_seq", "rise_fall_seq"]


def compute_user_baselines(all_features, session_names):
    """
    Computes each session's fused features once (cached for reuse in the
    main comparison loop), then for every user:
      - the std of num_blows/amp_rms_mean/duty_cycle across their own
        sessions (scalar baseline), and
      - the mean pairwise DTW distance between their own sessions' gap /
        duration / shape sequences (sequence baseline).
    Both fall back to a value pooled across all users when a user doesn't
    have enough of their own sessions to estimate it.
    """
    fused_cache = {}
    per_user_indices = {}

    for idx, feat in enumerate(all_features):
        fused = fused_features(feat['blow_features'], feat['session_features'])
        fused_cache[idx] = fused
        per_user_indices.setdefault(session_names[idx], []).append(idx)

    # ---- scalar baselines ----
    per_user_scalar_series = {key: {} for key in SCALAR_BASELINE_KEYS}
    for idx, fused in fused_cache.items():
        user = session_names[idx]
        values = {
            "num_blows": float(fused["num_blows"]),
            "amp_rms_mean": fused["amp_rms_mean"],
            "duty_cycle": fused["duty_cycle"],
        }
        for key in SCALAR_BASELINE_KEYS:
            per_user_scalar_series[key].setdefault(user, []).append(values[key])

    # Non-finite values (e.g. a zero-blow session's NaN amp_rms_mean) are
    # dropped here, not just guarded against below - otherwise a single
    # degenerate session would poison the *global* pool every user's
    # fallback depends on, not just that one user's own baseline.
    global_scalar_std = {}
    for key in SCALAR_BASELINE_KEYS:
        pooled = [v for vals in per_user_scalar_series[key].values() for v in vals if np.isfinite(v)]
        global_scalar_std[key] = float(np.std(pooled)) if len(pooled) > 1 else 1e-3

    # ---- DTW baselines: mean pairwise DTW distance among a user's own
    # sessions. O(sessions_per_user^2) DTW calls per user per key, but on
    # tiny sequences (a handful of blows) this is cheap even for hundreds
    # of users. ----
    per_user_dtw_dists = {key: {} for key in DTW_BASELINE_KEYS}
    for key in DTW_BASELINE_KEYS:
        for user, indices in per_user_indices.items():
            dists = []
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    seq1 = fused_cache[indices[a]][key]
                    seq2 = fused_cache[indices[b]][key]
                    d = dtw_distance(seq1, seq2)
                    # A pair involving a zero-blow session now returns inf
                    # (see dtw_distance) - exclude it from "this user's
                    # normal variability" rather than let it wreck the
                    # baseline scale for their other, legitimate sessions.
                    if np.isfinite(d):
                        dists.append(d)
            per_user_dtw_dists[key][user] = dists

    global_dtw_mean = {}
    for key in DTW_BASELINE_KEYS:
        pooled = [d for dists in per_user_dtw_dists[key].values() for d in dists]
        global_dtw_mean[key] = float(np.mean(pooled)) if pooled else 1e-3

    baselines = {}
    for user in session_names:
        if user in baselines:
            continue
        baseline = {}
        for key in SCALAR_BASELINE_KEYS:
            vals = [v for v in per_user_scalar_series[key][user] if np.isfinite(v)]
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            # `NaN < 1e-6` is always False (NaN comparisons never succeed),
            # so a NaN std would otherwise slip straight through this
            # fallback check instead of being caught by it.
            if not np.isfinite(std) or std < 1e-6:
                std = global_scalar_std[key] if global_scalar_std[key] > 1e-6 else 1e-3
            baseline[f"{key}_std"] = std
        for key in DTW_BASELINE_KEYS:
            dists = per_user_dtw_dists[key][user]
            mean_dist = float(np.mean(dists)) if dists else 0.0
            if not np.isfinite(mean_dist) or mean_dist < 1e-6:
                mean_dist = global_dtw_mean[key] if global_dtw_mean[key] > 1e-6 else 1e-3
            baseline[f"{key}_dtw_scale"] = mean_dist
        baselines[user] = baseline

    return fused_cache, baselines


def relative_z(enroll_val, attempt_val, baseline_std):
    enroll_val = float(enroll_val)
    attempt_val = float(attempt_val)
    if not (np.isfinite(enroll_val) and np.isfinite(attempt_val)):
        # A session with no measurable value for this feature (e.g. zero
        # detected blows -> NaN amp_rms_mean) isn't "close" or "far" from a
        # real pattern in any meaningful sense - treat it as maximally
        # dissimilar instead of letting NaN reach similarity_score(),
        # which would raise trying to round() it.
        return float("inf")
    return abs(enroll_val - attempt_val) / baseline_std


def similarity_score(z, decay=0.5):
    """
    Continuous 0-100 similarity score from a per-user z-score (or DTW
    distance expressed as a multiple of baseline), replacing named buckets
    (VERY_GOOD..BAD) that hide how close a case is to a boundary. 100 =
    matches this user's typical pattern exactly; the score decays smoothly
    (not in steps) as the attempt drifts further from that user's own
    normal variability. Rounded to the nearest 5 so the LLM only has to
    compare simple integers, not raw z-scores or decimals.
    """
    score = 100.0 * math.exp(-decay * z)
    return int(round(score / 5.0) * 5)


def cosine_position_score(cosine_sim, threshold):
    """
    Where this pair's raw embedding cosine similarity falls within the
    BORDERLINE band, on the same 0-100 scale as the other 6 scores: 100 =
    right at the threshold (a near-miss that would have been an
    auto-accept), 0 = at the far edge of the borderline band (a near-miss
    that would have been an auto-reject).

    Unlike similarity_score(), this is a plain linear interpolation, not
    an exponential decay from a per-user baseline — the borderline band is
    a fixed global width (BORDERLINE_MARGIN) by construction from the
    routing logic itself, so there's no per-user variability to normalize
    against here. Only ever called for pairs already routed to BORDERLINE.
    """
    margin = cosine_sim - threshold
    frac = (margin + BORDERLINE_MARGIN) / BORDERLINE_MARGIN
    frac = max(0.0, min(1.0, frac))  # clamp for float edge cases
    return int(round(frac * 100 / 5.0) * 5)


def convert_features_to_semantic_score_based(enroll_feat, attempt_feat, baseline, cosine_sim, threshold):
    """
    Reduce enroll vs. attempt into 7 0-100 similarity scores the LLM can
    reason about: how borderline the raw embedding match itself is, blow
    count, two scalar features (amplitude, mass) via per-user z-score, and
    three sequence features (gap, duration, shape) via per-user-relative
    DTW.
    """

    blow_z = relative_z(
        enroll_feat['num_blows'], attempt_feat['num_blows'],
        baseline['num_blows_std']
    )
    amplitude_z = relative_z(
        enroll_feat['amp_rms_mean'], attempt_feat['amp_rms_mean'],
        baseline['amp_rms_mean_std']
    )
    mass_z = relative_z(
        enroll_feat['duty_cycle'], attempt_feat['duty_cycle'],
        baseline['duty_cycle_std']
    )

    gap_z = dtw_relative_z(
        enroll_feat['gap_pattern'], attempt_feat['gap_pattern'],
        baseline['gap_pattern_dtw_scale']
    )
    duration_z = dtw_relative_z(
        enroll_feat['duration_seq'], attempt_feat['duration_seq'],
        baseline['duration_seq_dtw_scale']
    )
    shape_z = dtw_relative_z(
        enroll_feat['rise_fall_seq'], attempt_feat['rise_fall_seq'],
        baseline['rise_fall_seq_dtw_scale']
    )

    return {
        "cosine_score": cosine_position_score(cosine_sim, threshold),
        "blow_count_score": similarity_score(blow_z),
        "amplitude_score": similarity_score(amplitude_z),
        "mass_score": similarity_score(mass_z),
        "gap_score": similarity_score(gap_z),
        "duration_score": similarity_score(duration_z),
        "shape_score": similarity_score(shape_z),
        # raw z-scores / margin, kept for CSV/debug inspection; never shown to the LLM
        "cosine_margin": round(cosine_sim - threshold, 4),
        "blow_count_z": round(blow_z, 2),
        "amplitude_z": round(amplitude_z, 2),
        "mass_z": round(mass_z, 2),
        "gap_z": round(gap_z, 2),
        "duration_z": round(duration_z, 2),
        "shape_z": round(shape_z, 2),
    }


total_feat = 0
total_prompt = 0


def create_auth_prompt_score_based(enroll_feat, attempt_feat, cosine_sim, threshold, baseline):
    """
    Continuous scoring: every feature is a 0-100 similarity score, each
    normalized against this user's own per-session baseline (z-score for
    scalars, DTW-distance-vs-baseline for sequences) - see
    convert_features_to_semantic_score_based() / compute_user_baselines().
    """
    start_feat = time.time()
    semantic = convert_features_to_semantic_score_based(enroll_feat, attempt_feat, baseline, cosine_sim, threshold)
    end_feat = time.time()
    global total_feat
    total_feat += (end_feat - start_feat)

    start_prompts = time.time()
    prompt = f"""
You are a biometric verification assistant for blow audio patterns.

TASK:
Decide ACCEPT or REJECT for this blow pattern authentication attempt.

SIMILARITY SCORES (0-100 each; 100 = matches the enrolled pattern
exactly, 0 = no resemblance; each already accounts for this user's own
normal variability):

- embedding similarity: {semantic['cosine_score']}/100
- blow count similarity: {semantic['blow_count_score']}/100 (small
  differences can be a segmentation artifact, not necessarily a
  different person)
- gap pattern similarity: {semantic['gap_score']}/100
- duration pattern similarity: {semantic['duration_score']}/100
- shape pattern similarity: {semantic['shape_score']}/100
- amplitude (loudness) similarity: {semantic['amplitude_score']}/100
- blow "mass" (total effort) similarity: {semantic['mass_score']}/100

Weigh all scores together and decide ACCEPT or REJECT.

Return ONLY:
Decision: ACCEPT or REJECT
Confidence: 0-100
Reason: Describe Accept/Reject Decision
"""
    end_prompts = time.time()
    global total_prompt
    total_prompt += (end_prompts - start_prompts)
    return prompt


# =========================================================
# Rule-based (discrete-label) scoring — alternative to the continuous
# 0-100 scoring above. Instead of per-user z-scores/DTW-vs-baseline, each
# feature is thresholded against a fixed, global percentage-difference cutoff
# into one of five labels (VERY_GOOD/GOOD/OKAY/WEAK/BAD). No per-user
# baseline is needed here - this scoring system doesn't have one. Ported
# from newPrompt_6features.py (via newPrompt_6features_demo.py).
# =========================================================

def percentage_difference(a, b):
    """
    Symmetric percentage difference. Avoids division explosion for small
    values by averaging the two magnitudes in the denominator instead of
    dividing by either one alone.
    """
    a = float(a)
    b = float(b)

    denom = max((abs(a) + abs(b)) / 2.0, 1e-6)
    return abs(a - b) / denom


def diff_to_label(diff):
    """Convert a normalized percentage difference into a semantic label."""
    if diff <= 0.10:
        return "VERY_GOOD"
    elif diff <= 0.20:
        return "GOOD"
    elif diff <= 0.30:
        return "OKAY"
    elif diff <= 0.50:
        return "WEAK"
    else:
        return "BAD"


def cosine_rule_based_label(cosine_sim, threshold):
    """
    Rule-based label for the raw embedding similarity, on the same
    VERY_GOOD..BAD vocabulary as the other 6 features (unlike those, which
    threshold a computed percentage difference, this thresholds the
    cosine-vs-threshold margin directly):
      - VERY_GOOD: at or above the auto-accept threshold
      - BAD: at or below the far edge of the borderline band
        (threshold - BORDERLINE_MARGIN)
      - GOOD / OKAY / WEAK: the borderline band itself split into 3 equal
        sub-ranges, closest-to-threshold to closest-to-BAD, so an
        ambiguous case isn't collapsed into a single undifferentiated
        bucket
    """
    margin = cosine_sim - threshold
    if margin >= 0:
        return "VERY_GOOD"
    if margin <= -BORDERLINE_MARGIN:
        return "BAD"

    # Strictly inside the borderline band: frac goes from just above 0
    # (right at the threshold) to just below 1 (right at the BAD edge).
    frac = -margin / BORDERLINE_MARGIN
    if frac <= 1 / 3:
        return "GOOD"
    elif frac <= 2 / 3:
        return "OKAY"
    else:
        return "WEAK"


def sequence_similarity(seq1, seq2):
    """Compare two numeric sequences element-wise via percentage_difference."""
    n = min(len(seq1), len(seq2))

    if n == 0:
        return {"overall": "BAD", "details": []}

    labels = []
    for i in range(n):
        diff = percentage_difference(seq1[i], seq2[i])
        label = diff_to_label(diff)
        labels.append({
            "index": i,
            "enroll": round(seq1[i], 3),
            "attempt": round(seq2[i], 3),
            "diff": round(diff * 100, 1),
            "label": label
        })

    avg_diff = np.mean([percentage_difference(seq1[i], seq2[i]) for i in range(n)])
    overall = diff_to_label(avg_diff)

    return {"overall": overall, "details": labels}


def shape_diff_to_label(diff):
    # diff is a normalized percentage, 0.5 = 50% - shape (rise/fall ratio)
    # naturally varies more than the other features, so this uses wider
    # cutoffs than diff_to_label() rather than reusing it.
    if diff <= 0.50:
        return "VERY_GOOD"
    elif diff <= 1.00:
        return "GOOD"
    elif diff <= 1.50:
        return "OKAY"
    elif diff <= 2.00:
        return "WEAK"
    else:
        return "BAD"


def shape_similarity(shape1, shape2):
    n = min(len(shape1), len(shape2))

    if n == 0:
        return {"overall": "BAD", "details": []}

    labels = []
    diffs = []
    for i in range(n):
        s1 = float(shape1[i])
        s2 = float(shape2[i])
        denom = max((abs(s1) + abs(s2)) / 2.0, 1e-6)
        diff = abs(s1 - s2) / denom
        diffs.append(diff)
        label = shape_diff_to_label(diff)
        labels.append({
            "index": i,
            "enroll": round(s1, 3),
            "attempt": round(s2, 3),
            "diff_percent": round(diff * 100, 1),
            "label": label
        })

    avg_diff = np.mean(diffs)
    overall = shape_diff_to_label(avg_diff)

    return {"overall": overall, "details": labels}


def convert_features_to_semantic_rule_based(enroll_feat, attempt_feat, cosine_sim, threshold):
    """
    Reduce enroll vs. attempt into 7 discrete VERY_GOOD..BAD labels the LLM
    can reason about - the rule-based counterpart of
    convert_features_to_semantic_score_based() above, using fixed global
    percentage-difference cutoffs instead of per-user z-scores/DTW.
    """
    embedding_label = cosine_rule_based_label(cosine_sim, threshold)

    blow_diff = abs(enroll_feat['num_blows'] - attempt_feat['num_blows'])
    if blow_diff == 0:
        blow_label = "VERY_GOOD"
    elif blow_diff == 1:
        blow_label = "OKAY"
    else:
        blow_label = "BAD"

    gap_result = sequence_similarity(enroll_feat['gap_pattern'], attempt_feat['gap_pattern'])
    duration_result = sequence_similarity(enroll_feat['duration_seq'], attempt_feat['duration_seq'])
    shape_result = shape_similarity(enroll_feat['rise_fall_seq'], attempt_feat['rise_fall_seq'])

    amplitude_diff = percentage_difference(enroll_feat['amp_mean'], attempt_feat['amp_mean'])
    amplitude_label = diff_to_label(amplitude_diff)

    mass_diff = percentage_difference(enroll_feat['duty_cycle'], attempt_feat['duty_cycle'])
    mass_label = diff_to_label(mass_diff)

    return {
        "embedding_similarity": embedding_label,
        "blow_count_similarity": blow_label,
        "gap_similarity": gap_result["overall"],
        "duration_similarity": duration_result["overall"],
        "shape_similarity": shape_result["overall"],
        "amplitude_similarity": amplitude_label,
        "mass_similarity": mass_label,
        # per-blow breakdowns, kept for CSV/debug inspection; never shown to the LLM
        "shape_details": shape_result["details"],
        "gap_details": gap_result["details"],
        "duration_details": duration_result["details"],
    }


def create_auth_prompt_rule_based(enroll_feat, attempt_feat, cosine_sim, threshold, baseline=None):
    """
    Discrete scoring: every feature is a VERY_GOOD/GOOD/OKAY/WEAK/BAD label
    against a fixed global percentage-difference cutoff - no per-user
    baseline needed, so `baseline` is accepted (for a uniform call site
    alongside create_auth_prompt_score_based) but ignored.
    """
    start_feat = time.time()
    semantic = convert_features_to_semantic_rule_based(enroll_feat, attempt_feat, cosine_sim, threshold)
    end_feat = time.time()
    global total_feat
    total_feat += (end_feat - start_feat)

    start_prompts = time.time()
    prompt = f"""
You are a biometric verification assistant for blow audio patterns.

TASK:
Decide ACCEPT or REJECT for a blow pattern authentication attempt.

ENROLL vs ATTEMPT ANALYSIS:

- embedding similarity: {semantic['embedding_similarity']}
- blow count similarity: {semantic['blow_count_similarity']}
- gap similarity: {semantic['gap_similarity']}
- duration similarity: {semantic['duration_similarity']}
- shape similarity: {semantic['shape_similarity']}
- amplitude similarity: {semantic['amplitude_similarity']}
- mass similarity: {semantic['mass_similarity']}


- MUST be resolved into ACCEPT or REJECT based on the pattern similarity


Return ONLY:
Decision: ACCEPT or REJECT
Confidence: 0-100
Reason:
- Describe Accept/Reject Decision
"""
    end_prompts = time.time()
    global total_prompt
    total_prompt += (end_prompts - start_prompts)
    return prompt


# Which scoring method create_auth_prompt() below dispatches to - flip this
# to "rule_based" to switch the whole run over to the discrete-label scoring
# above instead of the continuous 0-100 scoring, with no other code changes
# needed (both are also included in build_run_signature() so a checkpoint
# from one scoring method is never silently resumed under the other).
SCORING_METHOD = "score_based"  # "score_based" or "rule_based"


def create_auth_prompt(enroll_feat, attempt_feat, cosine_sim, threshold, baseline):
    if SCORING_METHOD == "score_based":
        return create_auth_prompt_score_based(enroll_feat, attempt_feat, cosine_sim, threshold, baseline)
    elif SCORING_METHOD == "rule_based":
        return create_auth_prompt_rule_based(enroll_feat, attempt_feat, cosine_sim, threshold, baseline)
    else:
        raise ValueError(f"Unknown SCORING_METHOD: {SCORING_METHOD!r} (expected 'score_based' or 'rule_based')")


def parse_llm_response(text):
    decision = None
    confidence = None
    reason = None

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Decision:"):
            decision = line.split("Decision:")[1].strip()
        elif line.startswith("Confidence:"):
            confidence = line.split("Confidence:")[1].strip()
        elif line.startswith("Reason:"):
            reason = line.split("Reason:")[1].strip()

    return decision, confidence, reason


# =========================================================
# Checkpointing
# =========================================================

CHECKPOINT_FILE = "checkpoint_v4.json"
RESULTS_FILE = "results_v4.csv"
# Snapshot written only once a run fully finishes. RESULTS_FILE is truncated
# every time a fresh run starts (see csv_mode == "w" below), so this is the
# only place the results of the *last completed* run survive a later re-run.
RESULTS_FINAL_FILE = "results_v4_final.csv"
RESULTS_HEADER = [
    "pair", "user_i", "user_j", "cosine",
    "decision", "confidence", "result", "reason",
]

# How often to fsync the checkpoint + results file for pairs resolved by
# the cosine threshold alone (no LLM call). These are effectively free to
# redo after a crash, so they're batched to avoid an fsync on every one of
# the ~250k comparisons, which would otherwise dominate runtime on modest
# hardware. Pairs that *do* reach the LLM are always fsynced immediately,
# since those are the expensive ones actually worth protecting.
FAST_PATH_CHECKPOINT_INTERVAL = 50


def build_run_signature(num_users, sessions_per_user, k, q, n_features, n_embeddings):
    """
    Fingerprints the run configuration a checkpoint was produced under.
    Resuming against a different dataset size, different num_users/k/q, a
    different feature-set script version, or a different SCORING_METHOD
    would silently splice incompatible rows into the same results file, so
    this is checked before any checkpoint is trusted.
    """
    return {
        "feature_version": "v4_cosine_plus_6features",
        "feature_source": FEATURE_FILE_SUFFIX or "default",
        "scoring_method": SCORING_METHOD,
        "num_users": num_users,
        "sessions_per_user": sessions_per_user,
        "k": k,
        "q": q,
        "n_features": n_features,
        "n_embeddings": n_embeddings,
    }


def save_checkpoint(state):
    # Write to a temp file and atomically rename over the real one, so a
    # crash mid-write can never leave a half-written, unparseable
    # checkpoint behind (os.replace is atomic on POSIX and Windows).
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, CHECKPOINT_FILE)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: {CHECKPOINT_FILE} is unreadable/corrupt — ignoring it and starting fresh.")
        return None


if __name__ == "__main__":
    all_features = load_all_features_unformatted()
    all_embeddings = load_all_embeddings_unformatted()

    # all_features[i] and all_embeddings[i] must be the same physical
    # session - cosine_matrix[i] is compared against fused_cache[i] purely
    # by index below, with no name lookup at comparison time. Both lists
    # are sorted the same way (natural sort on the canonical session id),
    # so this should always hold for a complete LABBA_Data tree; catch it
    # here rather than silently computing metrics against misaligned data.
    if len(all_features) != len(all_embeddings):
        raise SystemExit(
            f"Feature/embedding count mismatch: {len(all_features)} feature "
            f"pairs vs {len(all_embeddings)} embeddings - check LABBA_Data "
            "for sessions missing a features pair or an embedding file."
        )
    for idx, (feat, emb) in enumerate(zip(all_features, all_embeddings)):
        if feat['filename'] != emb['filename']:
            raise SystemExit(
                f"all_features/all_embeddings are misaligned at index {idx} "
                f"({feat['filename']!r} vs {emb['filename']!r}) - fix the "
                "sort/filenames in LABBA_Data before trusting any metrics."
            )

    # Parameters
    num_users = 50
    sessions_per_user = 10  # 2 postures (Sit/Stand) x 5 sessions in LABBA_Data

    session_names = []
    for u in range(1, num_users + 1):
        session_names.extend([f"user{u}"] * sessions_per_user)

    # change the K and Q parameters accordingly to your need.
    k = 4
    q = 2

    run_signature = build_run_signature(
        num_users, sessions_per_user, k, q,
        len(all_features), len(all_embeddings)
    )

    checkpoint = load_checkpoint()
    if checkpoint is not None and checkpoint.get("signature") != run_signature:
        print(f"{CHECKPOINT_FILE} exists but was produced under a different run "
              "configuration (dataset size, num_users, k, q, feature-set "
              "version, or feature source changed) — ignoring it and starting "
              "fresh so results don't get mixed.")
        checkpoint = None

    if checkpoint is not None and checkpoint.get("completed"):
        TP, FP, TN, FN = checkpoint["TP"], checkpoint["FP"], checkpoint["TN"], checkpoint["FN"]
        # Backfill the snapshot if it's missing (e.g. a checkpoint from
        # before RESULTS_FINAL_FILE existed) so it's always available for a
        # completed run without needing a fresh run just to produce it.
        if os.path.exists(RESULTS_FILE) and not os.path.exists(RESULTS_FINAL_FILE):
            shutil.copyfile(RESULTS_FILE, RESULTS_FINAL_FILE)
        print(f"This configuration already completed a full run (see {RESULTS_FINAL_FILE}). "
              f"Delete {CHECKPOINT_FILE} to re-run from scratch — {RESULTS_FILE} will reset "
              f"automatically, but {RESULTS_FINAL_FILE} is left alone until the new run finishes.")
        print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
        print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
        print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)
        print("Accuracy:", (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0)
        raise SystemExit(0)

    cosine_matrix = compute_cosine_similarity_matrix(all_embeddings)
    distance_matrix = cosine_similarity_to_distance(cosine_matrix)

    user_thresholds = get_all_user_thresholds(distance_matrix, session_names, k, q)

    cosine_user_thresholds = {
        user: 1 - th
        for user, th in user_thresholds.items()
    }

    start_baseline = time.time()
    fused_cache, user_baselines = compute_user_baselines(all_features, session_names)
    baseline_time = time.time() - start_baseline

    if checkpoint is not None:
        resume_pair = (checkpoint["last_i"], checkpoint["last_j"])
        TP, FP, TN, FN = checkpoint["TP"], checkpoint["FP"], checkpoint["TN"], checkpoint["FN"]
        num_runs = checkpoint["num_runs"]
        total_time = checkpoint["total_time"]
        total_prompt = checkpoint["total_prompt"]
        total_feat = checkpoint["total_feat"]
        csv_mode = "a"
        print(f"Resuming after pair {resume_pair} — {num_runs} LLM call(s) already done "
              f"(TP={TP} FP={FP} TN={TN} FN={FN}).")
    else:
        resume_pair = None
        TP = FP = TN = FN = 0
        num_runs = 0
        total_time = 0.0
        total_prompt = 0.0
        total_feat = 0.0
        csv_mode = "w"

    np.fill_diagonal(cosine_matrix, -1)

    # Loaded lazily on the first pair that actually needs it, so a resume
    # that has nothing but cheap threshold-only pairs left doesn't have to
    # pay for loading the model at all.
    llm = None

    csv_file = open(RESULTS_FILE, csv_mode, newline="")
    writer = csv.writer(csv_file)
    if csv_mode == "w":
        writer.writerow(RESULTS_HEADER)
        csv_file.flush()
        os.fsync(csv_file.fileno())

    start_all = time.time()
    since_last_sync = 0

    try:
        for i in range(0, 500):
            for j in range(0, 500):
                if i == j:
                    continue
                if resume_pair is not None and (i, j) <= resume_pair:
                    continue

                session1 = i
                session2 = j
                cosine_sim = cosine_matrix[session1, session2]
                threshold = cosine_user_thresholds[session_names[session1]]
                margin = cosine_sim - threshold

                if cosine_sim >= threshold:
                    score_status = "ABOVE_THRESHOLD"
                elif margin > -BORDERLINE_MARGIN:
                    score_status = "BORDERLINE"
                else:
                    score_status = "BELOW_THRESHOLD"

                is_llm_pair = score_status == "BORDERLINE"

                if score_status == "ABOVE_THRESHOLD":
                    decision = "ACCEPT"
                    confidence = "100"
                    reason = "Cosine similarity is above the threshold."
                elif score_status == "BELOW_THRESHOLD":
                    decision = "REJECT"
                    confidence = "100"
                    reason = "Cosine similarity is below the threshold."

                else:
                    print(f"Comparing sessions {session1}  and {session2} ")
                    fuse_features_0 = fused_cache[session1]
                    fuse_features_1 = fused_cache[session2]
                    baseline = user_baselines[session_names[session1]]

                    prompt = create_auth_prompt(
                        enroll_feat=fuse_features_0,
                        attempt_feat=fuse_features_1,
                        cosine_sim=cosine_matrix[session1, session2],
                        threshold=cosine_user_thresholds[session_names[session1]],
                        baseline=baseline,
                    )
                    start_prompt = time.time()
                    chat_prompt = f"""<|system|>
                        You are a biometric reasoning assistant.
                        <|end|>

                        <|user|>
                        {prompt}
                        <|end|>

                        <|assistant|>
                        """
                    end_prompt = time.time()
                    total_prompt += (end_prompt - start_prompt)

                    if llm is None:
                        llm = Llama(
                            model_path=str(ASSETS_DIR / "Phi-3-mini-4k-instruct-q4.gguf"),
                            n_ctx=2048,
                            verbose=False
                        )

                    start = time.time()
                    response = llm(
                        chat_prompt,
                        max_tokens=512,
                        temperature=0.3,
                        stop=["<|end|>"]
                    )
                    end = time.time()

                    total_time += (end - start)
                    num_runs += 1
                    print(response["choices"][0]["text"])
                    print(f"Elapsed: {end - start:.3f} sec")
                    decision, confidence, reason = parse_llm_response(response["choices"][0]["text"])

                same_user = (session1 // sessions_per_user) == (session2 // sessions_per_user)

                if same_user:
                    if decision == "ACCEPT":
                        if is_llm_pair and not first_tp:
                            print(chat_prompt)
                            first_tp = True
                        TP += 1
                    else:
                        if is_llm_pair and not first_fn:
                            print(chat_prompt)
                            first_fn = True
                        FN += 1
                else:
                    if decision == "ACCEPT":
                        if is_llm_pair and not first_fp:
                            print(chat_prompt)
                            first_fp = True
                        FP += 1
                    else:
                        if is_llm_pair and not first_tn:
                            print(chat_prompt)
                            first_tn = True
                        TN += 1

                correct = (same_user and decision == "ACCEPT") or (not same_user and decision == "REJECT")
                writer.writerow([
                    f"{session1:03d}-{session2:03d}",
                    session_names[session1],
                    session_names[session2],
                    f"{cosine_matrix[session1, session2]:.3f}",
                    decision,
                    confidence,
                    "OK" if correct else "WRONG",
                    (reason or "").replace("\n", " ").replace("\r", "")
                ])

                since_last_sync += 1
                # Always durably checkpoint after an LLM call (expensive,
                # worth protecting); batch the cheap threshold-only pairs
                # so a crash there costs at most a short, fast replay.
                if is_llm_pair or since_last_sync >= FAST_PATH_CHECKPOINT_INTERVAL:
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
                    save_checkpoint({
                        "signature": run_signature,
                        "last_i": i,
                        "last_j": j,
                        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                        "num_runs": num_runs,
                        "total_time": total_time,
                        "total_prompt": total_prompt,
                        "total_feat": total_feat,
                        "completed": False,
                    })
                    since_last_sync = 0

        # Final sync so the last batch of fast-path pairs isn't left
        # uncheckpointed, then mark the run complete.
        csv_file.flush()
        os.fsync(csv_file.fileno())
        save_checkpoint({
            "signature": run_signature,
            "last_i": 499,
            "last_j": 499,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "num_runs": num_runs,
            "total_time": total_time,
            "total_prompt": total_prompt,
            "total_feat": total_feat,
            "completed": True,
        })

    except KeyboardInterrupt:
        print("\nInterrupted — progress through the last checkpoint is saved. "
              "Re-run the script to resume from there.")
        raise
    finally:
        csv_file.close()

    # Only reached if the loop finished normally (KeyboardInterrupt re-raises
    # above, so an interrupted run never gets here). RESULTS_FILE gets
    # truncated the next time a fresh run starts, so this snapshot is what
    # survives — untouched — until a future run completes and replaces it.
    shutil.copyfile(RESULTS_FILE, RESULTS_FINAL_FILE)
    print(f"Full results also saved to {RESULTS_FINAL_FILE}.")

    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)
    print("False Positive Rate:", FP / (TN + FP) if (TN + FP) > 0 else 0)
    print("False Negative Rate:", FN / (TP + FN) if (TP + FN) > 0 else 0)
    print("Accuracy:", (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0)

    end_all = time.time()
    print(f"Elapsed time this run: {end_all - start_all:.2f} seconds "
          "(TP/FP/TN/FN and timing totals above are cumulative across all resumes)")
    print(f"Baseline/fused-feature precompute time: {baseline_time:.2f} seconds "
          "(one-time, includes per-user pairwise DTW baselines, not per-pair)")
    average_time = total_time / num_runs if num_runs > 0 else 0
    print(f"\nAverage runtime Per Prompt: {average_time:.3f} sec over {num_runs} runs")
    average_time_prompt = total_prompt / num_runs if num_runs > 0 else 0
    print(f"\nAverage runtime Per PromptGenerated: {average_time_prompt:.10f} sec over {num_runs} runs")
    average_time_feat = total_feat / num_runs if num_runs > 0 else 0
    print(f"\nAverage runtime Per SemanticConversion: {average_time_feat:.8f} sec over {num_runs} runs")
