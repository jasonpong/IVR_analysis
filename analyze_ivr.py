import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import pandas as pd
import os
import glob
import wave
import audioop
import struct

def read_wav_file(wav_file):
    """
    Read WAV file, handling mu-law format using the working method.
    """
    try:
        # First, try reading as regular WAV with scipy
        try:
            sample_rate, audio = wavfile.read(wav_file)
            print(f"  Read as regular PCM WAV: {audio.dtype}, {audio.shape}")
            
            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0
            else:
                audio = audio.astype(np.float32)
                
            return sample_rate, audio
            
        except Exception as e:
            print(f"  Regular WAV read failed: {e}")
            print("  Using working mu-law conversion method...")
            
            # Use the exact working method you provided
            with open(wav_file, 'rb') as f:
                file_data = f.read()
            
            # Skip header (24 bytes for AU, but try different sizes for WAV)
            # Try 44 bytes first (standard WAV), then 24 bytes
            for header_size in [44, 24, 20, 16]:
                try:
                    audio_data = file_data[header_size:]
                    if len(audio_data) == 0:
                        continue
                        
                    print(f"  Trying header size: {header_size} bytes, data: {len(audio_data)} bytes")
                    
                    # Convert mu-law to linear using your exact method
                    linear_data = audioop.ulaw2lin(audio_data, 1)  # 1 byte input (mu-law is 8-bit)
                    
                    # Convert to numpy array
                    audio = np.frombuffer(linear_data, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                    
                    # Use 8kHz as in your working method
                    sample_rate = 8000
                    
                    print(f"  Success! Converted {len(audio)} samples at {sample_rate}Hz")
                    return sample_rate, audio
                    
                except Exception as convert_error:
                    print(f"    Header size {header_size} failed: {convert_error}")
                    continue
            
            raise Exception("All header sizes failed for mu-law conversion")
            
    except Exception as e:
        raise Exception(f"Could not read WAV file {wav_file}: {str(e)}")

def analyze_ticks(wav_file):
    """
    Analyze tick sounds in WAV file - UPDATED WITH CORRECT AUTOMATION DETECTION.
    
    Returns:
        dict with tick analysis results
    """
    try:
        print(f"Analyzing: {wav_file}")
        
        # Read the WAV file (handles both PCM and mu-law)
        sample_rate, audio = read_wav_file(wav_file)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            print(f"  Converted to mono")
        
        print(f"  Audio loaded: {len(audio)} samples, {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Get 8 seconds from 18 to 26 seconds (precise dialing period)
        start_sample = int(18 * sample_rate)  # Start at 18 seconds
        end_sample = min(len(audio), int(26 * sample_rate))  # End at 26 seconds (18 + 8)
        
        if start_sample >= len(audio):
            raise ValueError("Start time (18s) is beyond audio duration")
            
        audio = audio[start_sample:end_sample]
        analysis_duration = len(audio) / sample_rate
        
        print(f"  Using 18-26 second window ({analysis_duration:.1f}s total)")
        
        # Check if we have enough data
        if len(audio) < sample_rate:  # Less than 1 second
            raise ValueError("Audio too short (less than 1 second)")
        
        # Bandpass filter for 1100-1500 Hz
        nyquist = sample_rate / 2
        low_freq = 1100 / nyquist
        high_freq = 1500 / nyquist
        
        # Check if frequencies are valid
        if high_freq >= 1.0:
            raise ValueError(f"Sample rate {sample_rate}Hz too low for 1500Hz filter (Nyquist: {nyquist}Hz)")
        
        print(f"  Applying bandpass filter: 1100-1500 Hz")
        
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        # Get envelope using Hilbert transform
        envelope = np.abs(signal.hilbert(filtered_audio))
        
        # Smooth envelope (10ms window)
        window_size = max(1, int(0.01 * sample_rate))
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Dynamic threshold
        threshold = np.percentile(envelope_smooth, 85) * 2.0
        print(f"  Threshold: {threshold:.6f}")
        
        # Find events above threshold
        above_threshold = envelope_smooth > threshold
        
        # Find rising and falling edges
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Handle edge cases
        if len(above_threshold) > 0:
            if above_threshold[0]:
                starts = np.insert(starts, 0, 0)
            if above_threshold[-1]:
                ends = np.append(ends, len(above_threshold) - 1)
        
        print(f"  Found {len(starts)} potential events")
        
        # Initialize variables
        tick_times = []
        tick_candidates = []
        
        # Much tighter filtering based on waveform analysis
        min_duration_samples = max(1, int(0.005 * sample_rate))   # 5ms minimum (tighter)
        max_duration_samples = int(0.03 * sample_rate)           # 30ms maximum (much tighter - pulses are ~10-20ms)
        
        for start, end in zip(starts, ends):
            duration_samples = end - start
            duration_seconds = duration_samples / sample_rate
            
            if min_duration_samples <= duration_samples <= max_duration_samples:
                # Loosened tick characteristic checks
                segment_envelope = envelope_smooth[start:end]
                
                # Check for rise/fall (loosened requirements)
                rise_samples = min(5, len(segment_envelope) // 3)  # Reduced sample requirement
                fall_samples = min(5, len(segment_envelope) // 3)
                
                is_sharp = True  # Default to true, only check if we have enough samples
                if len(segment_envelope) > rise_samples + fall_samples:
                    rise_slope = (np.max(segment_envelope[:rise_samples]) - segment_envelope[0]) / rise_samples
                    fall_slope = (segment_envelope[-1] - np.max(segment_envelope[-fall_samples:])) / fall_samples
                    
                    # Much more lenient slope requirements
                    is_sharp = rise_slope > threshold * 0.05 or abs(fall_slope) > threshold * 0.05
                
                # Much stricter quiet period checks (baseline is very clean)
                quiet_window = int(0.01 * sample_rate)  # 10ms window (reduced)
                before_level = 0
                after_level = 0
                
                if start > quiet_window:
                    before_level = np.mean(envelope_smooth[start-quiet_window:start])
                    quiet_before = before_level < threshold * 0.2  # Much stricter (was 0.5)
                
                if end + quiet_window < len(envelope_smooth):
                    after_level = np.mean(envelope_smooth[end:end+quiet_window])
                    quiet_after = after_level < threshold * 0.2   # Much stricter (was 0.5)
                
                # Peak amplitude check (can be stricter - these pulses are very prominent)
                peak_amplitude = np.max(segment_envelope)
                is_prominent = peak_amplitude > threshold * 2.0  # Much stricter (was 1.2)
                
                # Scoring system (loosened)
                tick_score = 0
                score_details = []
                
                if is_sharp: 
                    tick_score += 1
                    score_details.append("sharp")
                if quiet_before: 
                    tick_score += 1
                    score_details.append("quiet_before")
                if quiet_after: 
                    tick_score += 1
                    score_details.append("quiet_after")
                if is_prominent: 
                    tick_score += 1
                    score_details.append("prominent")
                if duration_seconds < 0.04:  # Back to 40ms
                    tick_score += 1
                    score_details.append("short")
                
                # Accept if score >= 2 (back to working level)
                if tick_score >= 2:
                    tick_time = (start / sample_rate) + 18  # Add 18s offset for actual time
                    tick_times.append(tick_time)
                    
                    tick_candidates.append({
                        'time': tick_time,
                        'duration': duration_seconds,
                        'score': tick_score,
                        'score_details': score_details,
                        'amplitude': peak_amplitude,
                        'before_level': before_level,
                        'after_level': after_level,
                        'is_sharp': is_sharp,
                        'quiet_before': quiet_before,
                        'quiet_after': quiet_after,
                        'is_prominent': is_prominent
                    })
        
        print(f"  Valid ticks found: {len(tick_times)}")
        
        # Cluster ticks by timing consistency AND amplitude similarity
        if len(tick_candidates) > 16:
            print(f"  Found {len(tick_candidates)} candidates - clustering to find main sequence...")
            
            # Group ticks by similar amplitude (within 20% of each other)
            amplitude_clusters = []
            for candidate in tick_candidates:
                placed = False
                for cluster in amplitude_clusters:
                    cluster_mean_amp = np.mean([c['amplitude'] for c in cluster])
                    if abs(candidate['amplitude'] - cluster_mean_amp) / cluster_mean_amp < 0.2:  # 20% tolerance
                        cluster.append(candidate)
                        placed = True
                        break
                if not placed:
                    amplitude_clusters.append([candidate])
            
            # For each amplitude cluster, check timing consistency
            best_cluster = []
            best_score = 0
            
            for cluster in amplitude_clusters:
                if len(cluster) < 10:  # Need reasonable number of ticks
                    continue
                
                # Sort by time and calculate intervals
                cluster_sorted = sorted(cluster, key=lambda x: x['time'])
                cluster_times = [c['time'] for c in cluster_sorted]
                cluster_intervals = [cluster_times[i+1] - cluster_times[i] for i in range(len(cluster_times)-1)]
                
                if len(cluster_intervals) > 0:
                    # Score cluster by consistency and size
                    interval_cv = np.std(cluster_intervals) / np.mean(cluster_intervals)
                    amplitude_cv = np.std([c['amplitude'] for c in cluster_sorted]) / np.mean([c['amplitude'] for c in cluster_sorted])
                    
                    # Combined score: size, timing consistency, amplitude consistency
                    cluster_score = len(cluster) * (1.0 - min(interval_cv, 1.0)) * (1.0 - min(amplitude_cv, 1.0))
                    
                    print(f"    Cluster: {len(cluster)} ticks, interval CV: {interval_cv:.3f}, amplitude CV: {amplitude_cv:.3f}, score: {cluster_score:.1f}")
                    
                    if cluster_score > best_score:
                        best_score = cluster_score
                        best_cluster = cluster_sorted
            
            if best_cluster:
                print(f"  Selected cluster: {len(best_cluster)} ticks with score {best_score:.1f}")
                tick_candidates = best_cluster
                tick_times = [c['time'] for c in best_cluster]
            else:
                print(f"  No good cluster found, keeping all candidates")
        
        # Show details of first few ticks for debugging
        for i, tick in enumerate(tick_candidates[:5]):
            print(f"    Tick {i+1}: {tick['time']:.3f}s, score={tick['score']}/5, amp={tick['amplitude']:.0f}, criteria: {','.join(tick['score_details'])}")
        
        tick_times = sorted(tick_times)
        
        # Check if we have enough ticks for analysis
        if len(tick_times) < 2:
            return {
                'filename': os.path.basename(wav_file),
                'ticks_found': len(tick_times),
                'first_tick_time': tick_times[0] if len(tick_times) > 0 else None,
                'last_tick_time': None,
                'total_duration': None,
                'mean_interval': None,
                'std_interval': None,
                'min_interval': None,
                'max_interval': None,
                'assessment': 'INSUFFICIENT_DATA',
                'error': f'Only found {len(tick_times)} ticks (need at least 2)'
            }
        
        # Calculate intervals between consecutive ticks
        intervals = []
        for i in range(1, len(tick_times)):
            interval = tick_times[i] - tick_times[i-1]
            intervals.append(interval)
        
        intervals = np.array(intervals)
        
        # Calculate statistics
        if len(intervals) > 0:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            min_interval = np.min(intervals)
            max_interval = np.max(intervals)
        else:
            mean_interval = std_interval = min_interval = max_interval = 0
        
        # FIXED AUTOMATION ASSESSMENT - Based on your actual data patterns
        num_ticks = len(tick_times)
        
        # Calculate interval statistics if we have enough ticks
        if len(intervals) > 0:
            cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
        else:
            cv = 0
        
        # CORRECTED automation criteria based on your actual data patterns
        if num_ticks >= 15:  # 15+ ticks is clearly automated
            assessment = "AUTOMATED"
        elif num_ticks >= 12:  # 12-14 ticks likely automated  
            if cv < 1.5:  # Reasonable consistency (was too strict at 0.3)
                assessment = "LIKELY_AUTOMATED"
            else:
                assessment = "POSSIBLE_AUTOMATED"
        elif num_ticks >= 8:  # 8-11 ticks might be automated
            if cv < 1.0:  # Good consistency
                assessment = "POSSIBLE_AUTOMATED"
            else:
                assessment = "LIKELY_MANUAL"
        else:  # < 8 ticks
            assessment = "INSUFFICIENT_TICKS"
        
        print(f"  Analysis: {num_ticks} ticks, mean interval: {mean_interval:.3f}s, CV: {cv:.3f}, assessment: {assessment}")
        
        # Return results
        return {
            'filename': os.path.basename(wav_file),
            'ticks_found': len(tick_times),
            'first_tick_time': round(tick_times[0], 3),
            'last_tick_time': round(tick_times[-1], 3),
            'total_duration': round(tick_times[-1] - tick_times[0], 3),
            'mean_interval': round(mean_interval, 3),
            'std_interval': round(std_interval, 3),
            'min_interval': round(min_interval, 3),
            'max_interval': round(max_interval, 3),
            'assessment': assessment,
            'error': None
        }
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {
            'filename': os.path.basename(wav_file),
            'ticks_found': 0,
            'first_tick_time': None,
            'last_tick_time': None,
            'total_duration': None,
            'mean_interval': None,
            'std_interval': None,
            'min_interval': None,
            'max_interval': None,
            'assessment': 'ERROR',
            'error': str(e)
        }

def analyze_directory(directory_path, output_csv='tick_analysis_results.csv'):
    """
    Analyze all WAV files in a directory and save results to CSV.
    """
    # Find all WAV files
    wav_pattern = os.path.join(directory_path, "*.wav")
    wav_files = glob.glob(wav_pattern)
    
    if not wav_files:
        print(f"No WAV files found in {directory_path}")
        return None
    
    print(f"Found {len(wav_files)} WAV files")
    print("Processing files...")
    print("-" * 50)
    
    # Analyze each file
    results = []
    for i, wav_file in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] {os.path.basename(wav_file)}")
        result = analyze_ticks(wav_file)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n" + "=" * 50)
    print(f"Results saved to: {output_csv}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total files processed: {len(results)}")
    print(f"Files with errors: {len(df[df['error'].notna()])}")
    print(f"Automated calls: {len(df[df['assessment'] == 'AUTOMATED'])}")
    print(f"Likely automated: {len(df[df['assessment'] == 'LIKELY_AUTOMATED'])}")
    print(f"Possible automated: {len(df[df['assessment'] == 'POSSIBLE_AUTOMATED'])}")
    print(f"Manual calls: {len(df[df['assessment'] == 'LIKELY_MANUAL'])}")
    print(f"Insufficient data: {len(df[df['assessment'] == 'INSUFFICIENT_DATA'])}")
    
    return df

def print_summary_stats(csv_file):
    """
    Print summary statistics from the CSV results.
    """
    df = pd.read_csv(csv_file)
    
    # Filter out error cases for stats
    valid_df = df[(df['error'].isna()) & (df['ticks_found'] >= 2)]
    
    if len(valid_df) == 0:
        print("No valid results found for statistics")
        return
    
    print("\nTICK ANALYSIS SUMMARY STATISTICS:")
    print("=" * 50)
    print(f"Valid files analyzed: {len(valid_df)}")
    print(f"Average ticks found: {valid_df['ticks_found'].mean():.1f}")
    print(f"Average mean interval: {valid_df['mean_interval'].mean():.3f} seconds")
    print(f"Average total duration: {valid_df['total_duration'].mean():.3f} seconds")
    
    print("\nBY ASSESSMENT TYPE:")
    for assessment in valid_df['assessment'].unique():
        subset = valid_df[valid_df['assessment'] == assessment]
        if len(subset) > 0:
            print(f"\n{assessment} ({len(subset)} files):")
            print(f"  Avg ticks: {subset['ticks_found'].mean():.1f}")
            print(f"  Avg interval: {subset['mean_interval'].mean():.3f}s")
            print(f"  Avg duration: {subset['total_duration'].mean():.3f}s")

# Example usage:
# df = analyze_directory('/path/to/wav/files')
# print_summary_stats('tick_analysis_results.csv')
