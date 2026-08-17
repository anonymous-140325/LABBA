import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d


def read_audio_csv_blowprint(file_path, sr=48000):
    """
    Read audio signal from BlowPrint CSV format.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    sr : int
        Sample rate in Hz (default: 48000)
    
    Returns:
    --------
    signal : np.array
        Audio signal
    sample_rate : int
        Sample rate in Hz
    """
    df = pd.read_csv(file_path)
    all_samples = []

    # Parse semicolon-separated values
    for row in df[' Raw Audio Data']:
        samples = row.replace('"', '').split(';')
        all_samples.extend([float(val) for val in samples if val.strip() != ''])
    
    signal = np.array(all_samples)
    
    print(f"Loaded {len(signal)} samples from {file_path}")
    print(f"Duration: {len(signal)/sr:.2f} seconds at {sr} Hz")
    
    return signal, sr


def butter_bandpass_filter(signal, lowcut=80, highcut=2500, sr=48000, order=4):
    """
    Apply bandpass filter to remove noise outside typical blowing frequency range.
    
    Parameters adjusted for 48kHz sampling rate - can capture higher frequencies.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def compute_energy_envelope(signal, sr, window_ms=50):
    """
    Compute smooth energy envelope of the signal.
    
    Parameters:
    -----------
    signal : np.array
        Audio signal
    sr : int
        Sample rate
    window_ms : int
        Window size in milliseconds for smoothing
    
    Returns:
    --------
    envelope : np.array
        Smoothed energy envelope
    """
    # Compute RMS energy in short windows
    window_samples = int(sr * window_ms / 1000)
    
    # Square the signal (energy)
    energy = signal ** 2
    
    # Smooth with gaussian filter
    sigma = window_samples / 6  # Gaussian sigma for smoothing
    envelope = gaussian_filter1d(energy, sigma=sigma)
    
    # Convert to dB scale for better peak detection
    envelope = np.sqrt(envelope)  # Back to amplitude
    envelope = 20 * np.log10(envelope + 1e-10)  # dB scale
    
    return envelope

def extract_blow_features(signal, envelope, blow_regions, sr):
    features = []
    blow_start_times = []
    blow_end_times = []

    for i, (start, end) in enumerate(blow_regions):

        blow_sig = signal[start:end]
        blow_env = envelope[start:end]

        duration = (end - start) / sr
        start_time = start / sr
        end_time = end / sr

        # -----------------------------
        # Strength features
        # -----------------------------
        mean_amp = np.mean(np.abs(blow_sig))
        rms_amp = np.sqrt(np.mean(blow_sig**2))

        # -----------------------------
        # Stability features
        # -----------------------------
        std_amp = np.std(blow_env)
        cv_amp = std_amp / (np.mean(blow_env) + 1e-8)

        # -----------------------------
        # Shape features
        # -----------------------------
        peak_idx = np.argmax(blow_env)

        rise_time = peak_idx / sr
        fall_time = (len(blow_env) - peak_idx) / sr

        attack_slope = (blow_env[peak_idx] - blow_env[0]) / (rise_time + 1e-8)
        decay_slope = (blow_env[-1] - blow_env[peak_idx]) / (fall_time + 1e-8)

        features.append({
            "blow_id": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,

            "mean_amp": mean_amp,
            "rms_amp": rms_amp,

            "std_amp": std_amp,
            "cv_amp": cv_amp,

            "rise_time": rise_time,
            "fall_time": fall_time,
            "attack_slope": attack_slope,
            "decay_slope": decay_slope
        })

        blow_start_times.append(start_time)
        blow_end_times.append(end_time)

    # -----------------------------
    # Session-level features
    # -----------------------------
    if len(blow_start_times) > 0:
        start_delay = blow_start_times[0]

        blow_gaps = [
            blow_start_times[i + 1] - blow_end_times[i]
            for i in range(len(blow_start_times) - 1)
        ]

        session_duration = len(signal) / sr

        total_blow_duration = sum(
            f["duration"] for f in features
        )

    else:
        start_delay = 0
        blow_gaps = []
        session_duration = 0
        total_blow_duration = 0

    session_features = {
        "num_blows": len(blow_regions),

        # timing
        "start_delay": start_delay,
        "blow_gaps": blow_gaps,
        "session_duration": session_duration,

        # useful derived info
        "total_blow_duration": total_blow_duration,
        "duty_cycle": total_blow_duration / (session_duration + 1e-8)
    }

    return features, session_features


def detect_blow_instances(signal, sr=48000,
                          min_blow_duration_ms=150,
                          min_gap_ms=15,
                          fast_sigma_ms=8,
                          slow_sigma_ms=50,
                          plot = True):

    # =========================
    # STEP 0: PREPROCESS
    # =========================
    filtered_signal = butter_bandpass_filter(signal, sr=sr)
    rectified = np.abs(filtered_signal)
    

    fast_sigma = int(sr * fast_sigma_ms / 1000)
    slow_sigma = int(sr * slow_sigma_ms / 1000)

    # =========================
    # STEP 1: DUAL ENERGY REPRESENTATION
    # =========================
    energy_fast = gaussian_filter1d(rectified, sigma=fast_sigma)
    energy_slow = gaussian_filter1d(rectified, sigma=slow_sigma)
    energy_raw = gaussian_filter1d(rectified, sigma=1)  # minimal smoothing, for visualization only

    grad = np.diff(energy_fast, prepend = energy_fast[0])

    silence_thresh = 5e-6
    silence_duration = int(0.5 * sr)  # 500 ms (tune this)
    silence_count = 0


    found = False
    active_mask = np.zeros_like(grad, dtype=bool)
    last_false_idx = 0
    base = max(np.mean(energy_fast[:int(0.005*sr)]) * 1.2, 0.001)  # initial baseline from first 5ms
    for i, val in enumerate(grad):

        if abs(val) < silence_thresh:
            silence_count += 1
        else:
            silence_count = 0

        # reset condition
        if silence_count >= silence_duration:
            found = False
            last_false_idx = i

            start = max(0, i - silence_count + 1)
            end = i + 1

            active_mask[start:end] = False
            silence_count = 0
            continue


        if not found:
            if abs(val) > 5e-6:
                start = max(last_false_idx, i - int(0.3 * sr))
                base = max(np.mean(energy_fast[start:i]) * 1.2, 0.001)
                found = True
                active_mask[i] = True
            else:
                active_mask[i] = False

        else:
            if abs(val) > 5e-6 or energy_fast[i] > base:
                active_mask[i] = True
            else:
                active_mask[i] = False
                found = False
                last_false_idx = i

    
    min_samples = int(sr * min_blow_duration_ms / 1000)
    gap_samples = int(sr * min_gap_ms / 1000)


    i = 0
    while i < len(active_mask):
        if not active_mask[i]:
            gap_start = i
            
            # find end of False gap
            while i < len(active_mask) and not active_mask[i]:
                i += 1
            
            gap_end = i
            gap_len = gap_end - gap_start
            
            # if gap is small → fill it
            if gap_len <= gap_samples:
                active_mask[gap_start:gap_end] = True
        else:
            i += 1

    i = 0
    while i < len(active_mask):
        if active_mask[i]:
            start = i
            
            while i < len(active_mask) and active_mask[i]:
                i += 1
            
            end = i
            duration = end - start
            
            if duration < min_samples:
                active_mask[start:end] = False
        else:
            i += 1

    # slow signal
    grad_slow = np.diff(energy_slow, prepend = energy_slow[0])
    found = False
    active_mask_slow = np.zeros_like(grad_slow, dtype=bool)
    last_false_idx = 0
    silence_thresh = 4e-6
    silence_count = 0
    base = np.mean(energy_slow[:int(0.005*sr)])
    for i, val in enumerate(grad_slow):

        if abs(val) < silence_thresh and energy_slow[i] < base:
            silence_count += 1
        else:
            silence_count = 0

        # reset condition
        if silence_count >= silence_duration:
            found = False
            last_false_idx = i

            start = max(0, i - silence_count + 1)
            end = i + 1

            active_mask_slow[start:end] = False
            silence_count = 0
            continue

        if not found:
            if abs(val) > 4e-6:
                start = max(last_false_idx, i - int(0.2* sr))
                base = max(np.mean(energy_slow[start:i]) * 1.2, 0.005)
                found = True
                active_mask_slow[i] = True
            else:
                active_mask_slow[i] = False

        else:
            if abs(val) > 4e-6 or energy_slow[i] > base:
                active_mask_slow[i] = True
            else:
                active_mask_slow[i] = False
                found = False
                last_false_idx = i

    
    min_samples = int(sr * min_blow_duration_ms / 1000)
    gap_samples = int(sr * min_gap_ms / 1000)


    i = 0
    while i < len(active_mask_slow):
        if not active_mask_slow[i]:
            gap_start = i
            
            # find end of False gap
            while i < len(active_mask_slow) and not active_mask_slow[i]:
                i += 1
            
            gap_end = i
            gap_len = gap_end - gap_start
            
            # if gap is small → fill it
            if gap_len <= gap_samples:
                active_mask_slow[gap_start:gap_end] = True
        else:
            i += 1

    i = 0
    while i < len(active_mask_slow):
        if active_mask_slow[i]:
            start = i
            
            while i < len(active_mask_slow) and active_mask_slow[i]:
                i += 1
            
            end = i
            duration = end - start
            
            if duration < min_samples:
                active_mask_slow[start:end] = False
        else:
            i += 1

    final_mask = active_mask & active_mask_slow
    def mask_to_regions(mask):
        regions = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                regions.append((start, i))
                start = None
        if start is not None:
            regions.append((start, len(mask)))
        return regions
    final_regions = mask_to_regions(final_mask)

    merged_final = []
    min_gap_samples = int(sr * min_gap_ms / 1000)

    for region in final_regions:
        if not merged_final:
            merged_final.append(region)
        else:
            ps, pe = merged_final[-1]
            cs, ce = region

            if cs - pe < min_gap_samples:
                merged_final[-1] = (ps, ce)
            else:
                merged_final.append(region)

    valid_blows_final = []
    min_duration_samples = int(sr * min_blow_duration_ms / 1000)

    for s, e in merged_final:
        if (e - s) >= min_duration_samples:
            valid_blows_final.append((s, e))

    blow_features, session_features = extract_blow_features(
        signal=filtered_signal,
        envelope=energy_fast,
        blow_regions=valid_blows_final,
        sr=sr
    )

    if plot:
        time_axis = np.arange(len(signal)) / sr

        fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True)

        # =========================
        # RAW SIGNAL
        # =========================
        axes[0].plot(time_axis, signal, linewidth=0.5)
        axes[0].set_title("Raw Audio Signal")
        axes[0].grid(True, alpha=0.3)

        # =========================
        # FAST ENERGY + MASK
        # =========================
        axes[1].plot(time_axis, energy_fast, label="Fast Energy")

        start = None
        for i, v in enumerate(active_mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                axes[1].axvspan(start/sr, i/sr, color="green", alpha=0.25)
                start = None

        if start is not None:
            axes[1].axvspan(start/sr, len(active_mask)/sr, color="green", alpha=0.25)

        axes[1].set_title("FAST Gaussian Detector")

        # =========================
        # SLOW ENERGY + MASK
        # =========================
        axes[2].plot(time_axis, energy_slow, label="Slow Energy")

        start = None
        for i, v in enumerate(active_mask_slow):
            if v and start is None:
                start = i
            elif not v and start is not None:
                axes[2].axvspan(start/sr, i/sr, color="orange", alpha=0.25)
                start = None

        if start is not None:
            axes[2].axvspan(start/sr, len(active_mask_slow)/sr, color="orange", alpha=0.25)

        axes[2].set_title("SLOW Gaussian Detector")

        # =========================
        # FINAL FUSED RESULT
        # =========================
        axes[3].plot(time_axis, energy_raw, alpha=0.5, label="Fast Energy")
        

        for s, e in valid_blows_final:
            axes[3].axvspan(s/sr, e/sr, color="red", alpha=0.35)

        axes[3].set_title("FUSED (FAST ∩ SLOW) FINAL BLOWS")

        plt.tight_layout()
        plt.show()
    if plot:
        time_axis = np.arange(len(signal)) / sr

        # plotting style for publication figures
        LABEL_SIZE = 20
        TICK_SIZE = 14
        LINE_WIDTH = 2.2
        SPAN_ALPHA = 0.35

        # =========================
        # FAST DETECTOR - Separate Plot
        # =========================
        fig1, ax1 = plt.subplots(figsize=(16, 4.5))

        ax1.plot(
            time_axis,
            energy_fast,
            linewidth=LINE_WIDTH,
            color='#2E86AB',
            label='Fast Envelope'
        )

        start = None
        for i, v in enumerate(active_mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                ax1.axvspan(start/sr, i/sr, color="green", alpha=SPAN_ALPHA)
                start = None

        if start is not None:
            ax1.axvspan(start/sr, len(active_mask)/sr,
                        color="green", alpha=SPAN_ALPHA)

        ax1.set_xlabel("Time (s)", fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel("Amplitude Envelope", fontsize=LABEL_SIZE, fontweight='bold')

        ax1.tick_params(axis='both', labelsize=TICK_SIZE, width=1.5)
        ax1.grid(True, alpha=0.25, linestyle='--')

        ax1.set_xlim(0, len(signal) / sr)

        # Thicken axis borders
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig('fast_detector.png', bbox_inches='tight')
        plt.close()

        # =========================
        # SLOW DETECTOR - Separate Plot
        # =========================
        fig2, ax2 = plt.subplots(figsize=(16, 4.5))

        ax2.plot(
            time_axis,
            energy_slow,
            linewidth=LINE_WIDTH,
            color='#2E86AB',
            label='Slow Envelope'
        )

        start = None
        for i, v in enumerate(active_mask_slow):
            if v and start is None:
                start = i
            elif not v and start is not None:
                ax2.axvspan(start/sr, i/sr, color="orange", alpha=SPAN_ALPHA)
                start = None

        if start is not None:
            ax2.axvspan(start/sr, len(active_mask_slow)/sr,
                        color="orange", alpha=SPAN_ALPHA)

        ax2.set_xlabel("Time (s)", fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel("Amplitude Envelope", fontsize=LABEL_SIZE, fontweight='bold')

        ax2.tick_params(axis='both', labelsize=TICK_SIZE, width=1.5)
        ax2.grid(True, alpha=0.25, linestyle='--')

        ax2.set_xlim(0, len(signal) / sr)

        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig('slow_detector.png', bbox_inches='tight')
        plt.close()

        # =========================
        # FUSED OUTPUT - Separate Plot
        # =========================
        fig3, ax3 = plt.subplots(figsize=(16, 4.5))

        ax3.plot(
            time_axis,
            energy_fast,
            linewidth=LINE_WIDTH,
            color='#2E86AB',
            alpha=0.85,
            label='Envelope'
        )

        for s, e in valid_blows_final:
            ax3.axvspan(s/sr, e/sr, color="red", alpha=SPAN_ALPHA)

        ax3.set_xlabel("Time (s)", fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel("Amplitude Envelope", fontsize=LABEL_SIZE, fontweight='bold')

        ax3.tick_params(axis='both', labelsize=TICK_SIZE, width=1.5)
        ax3.grid(True, alpha=0.25, linestyle='--')

        ax3.set_xlim(0, len(signal) / sr)

        for spine in ax3.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig('fused_detector.png', bbox_inches='tight')
        plt.close()

    return blow_features, session_features


def analyze_blow_pattern(file_path, sr=48000, plot=True, save=False, **kwargs):
    """
    Complete pipeline to analyze blow pattern from BlowPrint CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing audio data
    sr : int
        Sample rate (default: 48000)
    plot : bool
        Whether to visualize
    **kwargs : dict
        Additional parameters for detect_blow_instances
    
    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    # Read audio
    signal, sample_rate = read_audio_csv_blowprint(file_path, sr=sr)
    
    # Detect blows
    blow_feature, session_feature = detect_blow_instances(
        signal, 
        sample_rate,
        plot=plot,
        **kwargs
    )
    import json
    from pathlib import Path
    input_path = Path(file_path)
    output_dir = input_path.parent
    base_name = input_path.stem

    blow_features_file = output_dir / f"{base_name}_blow_features.json"
    session_features_file = output_dir / f"{base_name}_session_features.json"

    if save:
        with open(blow_features_file, 'w') as f:
            json.dump(blow_feature, f, indent=4, default=lambda x: float(x))
        print(f"Saved blow features to: {blow_features_file}")

        with open(session_features_file, 'w') as f:
            json.dump(session_feature, f, indent=4, default=lambda x: float(x))
        print(f"Saved session features to: {session_features_file}")

        print("\nBlow-level features:")
        print(json.dumps(blow_feature, indent=4, default=lambda x: float(x)))

        print("\nSession-level features:")
        print(json.dumps(session_feature, indent=4, default=lambda x: float(x)))


def process_all_data():
    base_dir = "~/assets/BlowPrintData"
    base_dir = os.path.expanduser(base_dir)
    data_path = []
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if "RawAudio" in filename and "gfcc" not in filename and "mfcc" not in filename and ".csv" in filename:
                file_path = os.path.join(dirpath, filename)
                data_path.append(file_path)
    data_path.sort(key=lambda x: os.path.basename(x).lower())

    return data_path


if __name__ == "__main__":
    # Analyze a single file with 48kHz sample rate
    data_path = process_all_data()

    file_path = data_path[3]
    results = analyze_blow_pattern(
        file_path,
        sr=48000,
        plot=True,
        min_blow_duration_ms=100,
        save=False
    )
