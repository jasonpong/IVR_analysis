import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import describe
import warnings
warnings.filterwarnings('ignore')

def plot_audio_for_inspection(filepath, time_range=(0, None), figsize=(16, 8)):
    """
    Plot audio waveform and spectrogram for visual inspection to identify tick regions
    
    Parameters:
    -----------
    filepath : str
        Path to the mu-law WAV file
    time_range : tuple
        (start, end) in seconds, None for full file
    figsize : tuple
        Figure size for the plot
    """
    
    print(f"Loading: {filepath}")
    
    # Load the WAV file
    sample_rate, audio_data = wavfile.read(filepath)
    
    # Convert to float and normalize
    if audio_data.dtype == np.int8 or audio_data.dtype == np.uint8:
        audio_float = audio_data.astype(np.float32) / 128.0
    elif audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Handle stereo
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)
    
    # Time axis
    time = np.arange(len(audio_float)) / sample_rate
    
    # Apply time range if specified
    if time_range[0] is not None:
        start_idx = int(time_range[0] * sample_rate)
    else:
        start_idx = 0
    
    if time_range[1] is not None:
        end_idx = int(time_range[1] * sample_rate)
    else:
        end_idx = len(audio_float)
    
    audio_segment = audio_float[start_idx:end_idx]
    time_segment = time[start_idx:end_idx]
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1])
    
    # Plot 1: Waveform
    axes[0].plot(time_segment, audio_segment, 'b-', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Waveform - {filepath.split("/")[-1]}')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(time_segment[0], time_segment[-1])
    
    # Add time markers every second
    for i in range(int(time_segment[0]), int(time_segment[-1]) + 1):
        axes[0].axvline(x=i, color='gray', alpha=0.2, linestyle=':')
        if i % 5 == 0:
            axes[0].axvline(x=i, color='gray', alpha=0.4, linestyle='--')
    
    # Plot 2: Spectrogram
    f, t, Sxx = signal.spectrogram(audio_segment, sample_rate, 
                                   nperseg=min(512, len(audio_segment)), 
                                   noverlap=min(256, len(audio_segment)//2))
    t = t + time_segment[0]  # Adjust time offset
    
    pcm = axes[1].pcolormesh(t, f[f <= 4000], 10 * np.log10(Sxx[f <= 4000] + 1e-10), 
                             shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_title('Spectrogram (0-4000 Hz)')
    plt.colorbar(pcm, ax=axes[1], label='Power (dB)')
    axes[1].set_xlim(time_segment[0], time_segment[-1])
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 File Info:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio_float)/sample_rate:.2f} seconds")
    print(f"  Displayed range: {time_segment[0]:.2f}s - {time_segment[-1]:.2f}s")
    print(f"\n👀 Visual inspection: Identify where the 16 ticks occur")
    print(f"   Look for the vertical lines in both waveform and spectrogram")
    print(f"\n✏️  Once identified, run: analyze_tick_region(filepath, start_time, end_time)")
    
    return sample_rate


def analyze_tick_region(filepath, start_time, end_time, label="Unknown", show_plots=True):
    """
    Detailed analysis of a specific region containing ticks
    
    Parameters:
    -----------
    filepath : str
        Path to the mu-law WAV file
    start_time : float
        Start of tick region in seconds
    end_time : float
        End of tick region in seconds
    label : str
        Label for this sample ("Automated" or "Human")
    show_plots : bool
        Whether to display plots
    """
    
    print(f"\n{'='*80}")
    print(f"ANALYZING TICK REGION: {start_time:.2f}s - {end_time:.2f}s")
    print(f"File: {filepath.split('/')[-1]}")
    print(f"Label: {label}")
    print(f"{'='*80}")
    
    # Load the WAV file
    sample_rate, audio_data = wavfile.read(filepath)
    
    # Convert to float and normalize
    if audio_data.dtype == np.int8 or audio_data.dtype == np.uint8:
        audio_float = audio_data.astype(np.float32) / 128.0
    elif audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Handle stereo
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)
    
    # Extract the specified region
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    tick_region = audio_float[start_sample:end_sample]
    time_region = np.arange(len(tick_region)) / sample_rate + start_time
    
    # Multiple detection methods
    results = {}
    
    # Method 1: Amplitude threshold
    abs_audio = np.abs(tick_region)
    thresholds = [90, 95, 97, 99, 99.5, 99.9]
    
    print(f"\n🎯 PEAK DETECTION RESULTS:")
    best_threshold = None
    best_peaks = None
    
    for thresh_pct in thresholds:
        threshold = np.percentile(abs_audio, thresh_pct)
        peaks, properties = signal.find_peaks(
            abs_audio,
            height=threshold,
            distance=int(0.05 * sample_rate)  # Min 50ms between peaks
        )
        print(f"  {thresh_pct:5.1f}% threshold: {len(peaks):2d} peaks detected (threshold={threshold:.4f})")
        
        # If we get close to 16 peaks, this might be our best threshold
        if best_peaks is None or abs(len(peaks) - 16) < abs(len(best_peaks) - 16):
            best_threshold = threshold
            best_peaks = peaks
    
    # Use the best threshold for analysis
    peaks = best_peaks
    print(f"\n  ➜ Using {len(peaks)} peaks for analysis")
    
    # Calculate tick characteristics
    if len(peaks) > 0:
        print(f"\n📏 TICK CHARACTERISTICS:")
        
        # Amplitude analysis
        peak_amplitudes = abs_audio[peaks]
        print(f"  Amplitude:")
        print(f"    Mean: {np.mean(peak_amplitudes):.4f}")
        print(f"    Std Dev: {np.std(peak_amplitudes):.4f}")
        print(f"    CV: {np.std(peak_amplitudes)/np.mean(peak_amplitudes):.4f}")
        
        # Duration analysis (width of each tick)
        tick_widths = []
        for peak in peaks:
            # Find where amplitude drops to 50% of peak value
            half_max = abs_audio[peak] * 0.5
            left_idx = peak
            right_idx = peak
            
            while left_idx > 0 and abs_audio[left_idx] > half_max:
                left_idx -= 1
            while right_idx < len(abs_audio) - 1 and abs_audio[right_idx] > half_max:
                right_idx += 1
            
            width_ms = (right_idx - left_idx) * 1000 / sample_rate
            tick_widths.append(width_ms)
        
        if tick_widths:
            print(f"  Tick Width (ms):")
            print(f"    Mean: {np.mean(tick_widths):.2f}")
            print(f"    Std Dev: {np.std(tick_widths):.2f}")
    
    # Interval analysis
    if len(peaks) > 1:
        print(f"\n⏱️  INTERVAL ANALYSIS:")
        
        intervals_ms = np.diff(peaks) * 1000 / sample_rate
        
        print(f"  Basic Statistics:")
        print(f"    Count: {len(intervals_ms)} intervals")
        print(f"    Mean: {np.mean(intervals_ms):.2f} ms")
        print(f"    Median: {np.median(intervals_ms):.2f} ms")
        print(f"    Std Dev: {np.std(intervals_ms):.2f} ms")
        print(f"    Min: {np.min(intervals_ms):.2f} ms")
        print(f"    Max: {np.max(intervals_ms):.2f} ms")
        print(f"    Range: {np.max(intervals_ms) - np.min(intervals_ms):.2f} ms")
        
        # Key metric for automation detection
        cv = np.std(intervals_ms) / np.mean(intervals_ms)
        print(f"\n  🔑 KEY METRIC:")
        print(f"    Coefficient of Variation (CV): {cv:.4f}")
        
        # Classification
        if cv < 0.05:
            classification = "STRONGLY AUTOMATED"
            confidence = 95
        elif cv < 0.10:
            classification = "LIKELY AUTOMATED"
            confidence = 80
        elif cv < 0.20:
            classification = "UNCERTAIN"
            confidence = 50
        elif cv < 0.30:
            classification = "LIKELY HUMAN"
            confidence = 80
        else:
            classification = "STRONGLY HUMAN"
            confidence = 95
        
        print(f"\n  🤖 CLASSIFICATION:")
        print(f"    Prediction: {classification}")
        print(f"    Confidence: {confidence}%")
        print(f"    Actual Label: {label}")
        print(f"    {'✅ CORRECT' if (label.upper() in classification) else '❌ INCORRECT' if label != 'Unknown' else ''}")
        
        # Pattern analysis
        print(f"\n  📊 PATTERN ANALYSIS:")
        
        # Check for grouping (human behavior)
        long_intervals = intervals_ms > (np.mean(intervals_ms) + 1.5 * np.std(intervals_ms))
        if np.any(long_intervals):
            print(f"    Grouping detected: Yes (at positions {np.where(long_intervals)[0] + 1})")
        else:
            print(f"    Grouping detected: No")
        
        # Check for drift (gradual speed change)
        if len(intervals_ms) > 5:
            first_half = np.mean(intervals_ms[:len(intervals_ms)//2])
            second_half = np.mean(intervals_ms[len(intervals_ms)//2:])
            drift = (second_half - first_half) / first_half * 100
            print(f"    Timing drift: {drift:+.1f}%")
        
        # Regularity score
        consecutive_changes = np.abs(np.diff(intervals_ms))
        mean_change = np.mean(consecutive_changes)
        print(f"    Mean consecutive change: {mean_change:.2f} ms")
        
        # Save results
        results = {
            'file': filepath.split('/')[-1],
            'region': (start_time, end_time),
            'label': label,
            'num_ticks': len(peaks),
            'intervals_ms': intervals_ms,
            'cv': cv,
            'mean_interval': np.mean(intervals_ms),
            'std_interval': np.std(intervals_ms),
            'classification': classification,
            'confidence': confidence
        }
    
    # Visualization
    if show_plots and len(peaks) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Waveform with detected ticks
        axes[0].plot(time_region, tick_region, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].plot(time_region[peaks], tick_region[peaks], 'ro', markersize=8, 
                    label=f'{len(peaks)} ticks detected')
        axes[0].axhline(y=best_threshold, color='r', linestyle='--', alpha=0.5, 
                       label=f'Threshold')
        axes[0].axhline(y=-best_threshold, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{label} Sample: {filepath.split("/")[-1]} ({start_time:.1f}s - {end_time:.1f}s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Interval histogram
        if len(peaks) > 1:
            axes[1].hist(intervals_ms, bins=min(20, len(intervals_ms)), 
                        edgecolor='black', alpha=0.7, color='blue')
            axes[1].axvline(np.mean(intervals_ms), color='r', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(intervals_ms):.1f}ms')
            axes[1].axvline(np.median(intervals_ms), color='orange', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(intervals_ms):.1f}ms')
            axes[1].set_xlabel('Interval (ms)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Interval Distribution (CV = {cv:.3f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Interval sequence
            axes[2].plot(range(1, len(intervals_ms) + 1), intervals_ms, 'o-', 
                        markersize=8, linewidth=2)
            axes[2].axhline(np.mean(intervals_ms), color='r', linestyle='--', 
                          alpha=0.5, label='Mean')
            axes[2].fill_between(range(1, len(intervals_ms) + 1),
                                np.mean(intervals_ms) - np.std(intervals_ms),
                                np.mean(intervals_ms) + np.std(intervals_ms),
                                alpha=0.2, color='red', label='±1 STD')
            axes[2].set_xlabel('Interval Number')
            axes[2].set_ylabel('Duration (ms)')
            axes[2].set_title(f'Interval Sequence - Classification: {classification} ({confidence}%)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Not enough ticks for interval analysis', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[2].text(0.5, 0.5, 'Not enough ticks for interval analysis', 
                        ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    return results


# Batch analysis function
def analyze_multiple_regions(analyses):
    """
    Compare multiple analyzed regions
    
    Parameters:
    -----------
    analyses : list of tuples
        [(filepath, start_time, end_time, label), ...]
    """
    
    all_results = []
    
    for filepath, start, end, label in analyses:
        result = analyze_tick_region(filepath, start, end, label, show_plots=False)
        if result:
            all_results.append(result)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'File':<30} {'Label':<10} {'Ticks':<6} {'CV':<8} {'Mean Int':<10} {'Classification':<20}")
    print(f"{'-'*30} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*20}")
    
    for r in all_results:
        print(f"{r['file'][:30]:<30} {r['label']:<10} {r['num_ticks']:<6} "
              f"{r['cv']:<8.4f} {r['mean_interval']:<10.2f} {r['classification']:<20}")
    
    # Separate by label
    automated = [r for r in all_results if 'AUTO' in r['label'].upper()]
    human = [r for r in all_results if 'HUMAN' in r['label'].upper() or 'MANUAL' in r['label'].upper()]
    
    if automated:
        print(f"\n📊 AUTOMATED SAMPLES:")
        cvs = [r['cv'] for r in automated]
        print(f"  CV range: {min(cvs):.4f} - {max(cvs):.4f}")
        print(f"  CV mean: {np.mean(cvs):.4f}")
    
    if human:
        print(f"\n👤 HUMAN SAMPLES:")
        cvs = [r['cv'] for r in human]
        print(f"  CV range: {min(cvs):.4f} - {max(cvs):.4f}")
        print(f"  CV mean: {np.mean(cvs):.4f}")
    
    return all_results


# Usage examples:
print("📌 USAGE INSTRUCTIONS:")
print("="*50)
print("\nStep 1: Plot audio to identify tick regions")
print("  >>> plot_audio_for_inspection('file.wav')")
print("  >>> plot_audio_for_inspection('file.wav', time_range=(5, 20))  # Zoom in")
print("\nStep 2: Analyze specific tick region")
print("  >>> analyze_tick_region('file.wav', start_time=7.5, end_time=9.5, label='Human')")
print("\nStep 3: Batch analysis (optional)")
print("  >>> analyses = [")
print("        ('file1.wav', 7.5, 9.5, 'Human'),")
print("        ('file2.wav', 10.0, 13.0, 'Automated'),")
print("      ]")
print("  >>> results = analyze_multiple_regions(analyses)")
