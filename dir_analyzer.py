#!/usr/bin/env python3
"""
Batch IVR Tick Analyzer
Analyzes multiple audio files for automated vs human input patterns
Outputs results to CSV file
"""

import numpy as np
import scipy.signal as signal
from scipy.cluster.vq import kmeans2, whiten
import audioop
import struct
import os
import glob
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_mulaw_wav(filepath):
    """Load mu-law encoded WAV file"""
    try:
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        # Find the 'data' chunk
        data_start = file_data.find(b'data')
        if data_start == -1:
            # Try fixed header sizes
            for header_size in [44, 58, 24, 20, 16]:
                try:
                    audio_data = file_data[header_size:]
                    if len(audio_data) == 0:
                        continue
                    
                    # Convert mu-law to linear
                    linear_data = audioop.ulaw2lin(audio_data, 1)
                    audio = np.frombuffer(linear_data, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                    sample_rate = 8000
                    
                    return sample_rate, audio
                except:
                    continue
        else:
            # Found 'data' chunk
            data_size_start = data_start + 4
            data_size = struct.unpack('<I', file_data[data_size_start:data_size_start + 4])[0]
            audio_data_start = data_start + 8
            audio_data = file_data[audio_data_start:audio_data_start + data_size]
            
            # Convert mu-law to linear
            linear_data = audioop.ulaw2lin(audio_data, 1)
            audio = np.frombuffer(linear_data, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            sample_rate = 8000
            
            return sample_rate, audio
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def detect_ticks(audio_region, sample_rate):
    """Detect tick sounds in audio region using derivative method"""
    
    # Method 1: Derivative-based detection
    derivative = np.abs(np.diff(audio_region))
    
    # Try different thresholds to find ~16 ticks
    best_peaks = None
    for percentile in [99, 99.5, 99.7, 99.9]:
        threshold = np.percentile(derivative, percentile)
        peaks, _ = signal.find_peaks(
            derivative,
            height=threshold,
            distance=int(0.05 * sample_rate)  # Min 50ms between ticks
        )
        
        if best_peaks is None or abs(len(peaks) - 16) < abs(len(best_peaks) - 16):
            best_peaks = peaks
            
        if len(peaks) == 16:  # Perfect match
            break
    
    # Validate amplitude consistency
    if len(best_peaks) > 0:
        peak_amplitudes = np.abs(audio_region[best_peaks])
        median_amp = np.median(peak_amplitudes)
        
        # Filter peaks by amplitude consistency
        final_peaks = []
        for i, peak in enumerate(best_peaks):
            if 0.5 * median_amp <= peak_amplitudes[i] <= 2.0 * median_amp:
                final_peaks.append(peak)
        
        return np.array(final_peaks)
    
    return best_peaks


def analyze_intervals(peaks, sample_rate):
    """Analyze intervals between detected ticks"""
    
    if len(peaks) < 2:
        return {
            'num_ticks': len(peaks),
            'cv': None,
            'classification': 'INSUFFICIENT_DATA',
            'confidence': 0,
            'key_parameter': f'Only {len(peaks)} ticks detected'
        }
    
    # Calculate intervals in milliseconds
    intervals_ms = np.diff(peaks) * 1000 / sample_rate
    
    # Basic statistics
    cv = np.std(intervals_ms) / np.mean(intervals_ms)
    mean_interval = np.mean(intervals_ms)
    std_interval = np.std(intervals_ms)
    
    # Mean consecutive change
    consecutive_changes = np.abs(np.diff(intervals_ms))
    mean_consecutive_change = np.mean(consecutive_changes) if len(consecutive_changes) > 0 else 0
    
    # Check for bimodal distribution
    has_bimodal = False
    cluster_consistency = 1.0
    
    if len(intervals_ms) >= 8:
        try:
            # K-means clustering to find two groups
            normalized = whiten(intervals_ms.reshape(-1, 1)).flatten()
            centroids, labels = kmeans2(normalized, 2, minit='points')
            
            cluster_0 = intervals_ms[labels == 0]
            cluster_1 = intervals_ms[labels == 1]
            
            if len(cluster_0) > 1 and len(cluster_1) > 1:
                cv_cluster_0 = np.std(cluster_0) / np.mean(cluster_0) if np.mean(cluster_0) > 0 else 1
                cv_cluster_1 = np.std(cluster_1) / np.mean(cluster_1) if np.mean(cluster_1) > 0 else 1
                separation = abs(np.mean(cluster_0) - np.mean(cluster_1))
                
                if separation > 20 and (cv_cluster_0 < 0.1 or cv_cluster_1 < 0.1):
                    has_bimodal = True
                    cluster_consistency = min(cv_cluster_0, cv_cluster_1)
        except:
            pass
    
    # Check for periodic pattern
    periodic_pattern = False
    if len(intervals_ms) > 6:
        long_intervals = intervals_ms > (np.mean(intervals_ms) + 0.3 * np.std(intervals_ms))
        long_positions = np.where(long_intervals)[0]
        
        if len(long_positions) >= 3:
            gaps = np.diff(long_positions)
            if len(gaps) > 0:
                gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 1
                periodic_pattern = gap_cv < 0.5
    
    # Classification logic
    key_parameter = ""
    
    if cv < 0.05:
        classification = "STRONGLY_AUTOMATED"
        confidence = 95
        key_parameter = f"CV={cv:.3f} (extremely regular)"
        
    elif has_bimodal and cluster_consistency < 0.1:
        classification = "AUTOMATED_WITH_PAUSES"
        confidence = 90
        key_parameter = f"Bimodal with cluster_CV={cluster_consistency:.3f}"
        
    elif cv < 0.20:
        if mean_consecutive_change < 50 or periodic_pattern:
            classification = "AUTOMATED"
            confidence = 85
            key_parameter = f"CV={cv:.3f}, mean_change={mean_consecutive_change:.1f}ms"
        else:
            classification = "LIKELY_AUTOMATED"
            confidence = 75
            key_parameter = f"CV={cv:.3f}"
            
    elif cv < 0.30:
        if mean_consecutive_change < 40:
            classification = "LIKELY_AUTOMATED"
            confidence = 70
            key_parameter = f"CV={cv:.3f}, mean_change={mean_consecutive_change:.1f}ms"
        else:
            classification = "UNCERTAIN"
            confidence = 50
            key_parameter = f"CV={cv:.3f} (borderline)"
            
    elif cv < 0.40:
        classification = "LIKELY_HUMAN"
        confidence = 75
        key_parameter = f"CV={cv:.3f} (variable timing)"
        
    else:
        classification = "STRONGLY_HUMAN"
        confidence = 90
        key_parameter = f"CV={cv:.3f} (highly variable)"
    
    # Override for perfect bimodal
    if has_bimodal and cluster_consistency < 0.05:
        classification = "STRONGLY_AUTOMATED_BIMODAL"
        confidence = 95
        key_parameter = f"Perfect bimodal, cluster_CV={cluster_consistency:.3f}"
    
    return {
        'num_ticks': len(peaks),
        'cv': cv,
        'mean_interval': mean_interval,
        'std_interval': std_interval,
        'mean_consecutive_change': mean_consecutive_change,
        'has_bimodal': has_bimodal,
        'periodic_pattern': periodic_pattern,
        'classification': classification,
        'confidence': confidence,
        'key_parameter': key_parameter
    }


def process_file(filepath, start_time, end_time):
    """Process a single audio file"""
    
    # Load audio
    sample_rate, audio = load_mulaw_wav(filepath)
    
    if audio is None:
        return {
            'filename': os.path.basename(filepath),
            'classification': 'ERROR',
            'confidence': 0,
            'num_ticks': 0,
            'cv': None,
            'mean_interval_ms': None,
            'key_parameter': 'Failed to load audio'
        }
    
    # Extract region of interest
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Ensure we don't exceed audio length
    end_sample = min(end_sample, len(audio))
    
    if start_sample >= len(audio):
        return {
            'filename': os.path.basename(filepath),
            'classification': 'ERROR',
            'confidence': 0,
            'num_ticks': 0,
            'cv': None,
            'mean_interval_ms': None,
            'key_parameter': 'Start time beyond audio length'
        }
    
    audio_region = audio[start_sample:end_sample]
    
    # Detect ticks
    peaks = detect_ticks(audio_region, sample_rate)
    
    # Analyze intervals
    results = analyze_intervals(peaks, sample_rate)
    
    return {
        'filename': os.path.basename(filepath),
        'classification': results['classification'],
        'confidence': results['confidence'],
        'num_ticks': results['num_ticks'],
        'cv': results['cv'],
        'mean_interval_ms': results.get('mean_interval', None),
        'key_parameter': results['key_parameter']
    }


def batch_analyze(directory, start_time, end_time, output_file='results.csv'):
    """
    Analyze all WAV files in a directory
    
    Parameters:
    -----------
    directory : str
        Path to directory containing WAV files
    start_time : float
        Start time in seconds for analysis window
    end_time : float
        End time in seconds for analysis window
    output_file : str
        Output CSV filename
    """
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    
    if not wav_files:
        print(f"No WAV files found in {directory}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    print(f"Analysis window: {start_time:.1f}s - {end_time:.1f}s")
    print("-" * 60)
    
    # Process each file
    all_results = []
    
    for i, filepath in enumerate(wav_files, 1):
        print(f"Processing {i}/{len(wav_files)}: {os.path.basename(filepath)}")
        
        result = process_file(filepath, start_time, end_time)
        all_results.append(result)
        
        # Print summary
        print(f"  → {result['classification']} (confidence: {result['confidence']}%)")
        print(f"  → {result['num_ticks']} ticks detected")
        if result['cv']:
            print(f"  → CV: {result['cv']:.3f}")
        print(f"  → Key: {result['key_parameter']}")
        print()
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'classification', 'confidence', 'num_ticks', 
                     'cv', 'mean_interval_ms', 'key_parameter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print("-" * 60)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    classifications = [r['classification'] for r in all_results]
    print("\nSummary:")
    for cls in set(classifications):
        count = classifications.count(cls)
        percentage = count / len(classifications) * 100
        print(f"  {cls}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python batch_ivr_analyzer.py <directory> <start_time> <end_time>")
        print("Example: python batch_ivr_analyzer.py ./recordings 10.0 13.0")
        sys.exit(1)
    
    directory = sys.argv[1]
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])
    
    batch_analyze(directory, start_time, end_time)
