import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import pandas as pd
import os
import glob
import audioop

def read_mulaw_wav(wav_file):
    """Read mu-law WAV file"""
    try:
        sample_rate, audio = wavfile.read(wav_file)
        if audio.dtype == np.int16:
            return sample_rate, audio.astype(np.float32) / 32768.0
    except:
        # Mu-law conversion
        with open(wav_file, 'rb') as f:
            data = f.read()
        for header_size in [44, 24]:
            try:
                audio_data = data[header_size:]
                linear_data = audioop.ulaw2lin(audio_data, 1)
                audio = np.frombuffer(linear_data, dtype=np.int16).astype(np.float32) / 32768.0
                return 8000, audio
            except:
                continue
    raise Exception("Could not read file")

def detect_ticks(wav_file):
    """Simple tick detection and automation assessment"""
    
    # Read audio
    sample_rate, audio = read_mulaw_wav(wav_file)
    
    # Extract 18-26 second window
    start = int(18 * sample_rate)
    end = int(26 * sample_rate)
    if start >= len(audio):
        return {'filename': os.path.basename(wav_file), 'error': 'File too short'}
    
    audio = audio[start:min(end, len(audio))]
    
    # Bandpass filter 1100-1500 Hz
    nyquist = sample_rate / 2
    b, a = signal.butter(4, [1100/nyquist, min(1500/nyquist, 0.99)], btype='band')
    filtered = signal.filtfilt(b, a, audio)
    
    # Find peaks
    envelope = np.abs(signal.hilbert(filtered))
    threshold = np.percentile(envelope, 90) * 1.5
    
    # Initial peak detection
    peaks, _ = signal.find_peaks(envelope, height=threshold, distance=int(0.01*sample_rate))
    
    # Filter peaks by tick duration (10-50ms)
    valid_peaks = []
    for peak in peaks:
        # Find pulse boundaries (where signal drops below threshold)
        pulse_start = peak
        pulse_end = peak
        
        # Find start of pulse (working backwards)
        for i in range(peak, max(0, peak - int(0.1*sample_rate)), -1):
            if envelope[i] < threshold * 0.5:  # 50% of threshold
                pulse_start = i
                break
        
        # Find end of pulse (working forwards)  
        for i in range(peak, min(len(envelope), peak + int(0.1*sample_rate))):
            if envelope[i] < threshold * 0.5:  # 50% of threshold
                pulse_end = i
                break
        
        # Check if pulse duration is tick-like (10-50ms)
        pulse_duration = (pulse_end - pulse_start) / sample_rate
        if 0.01 <= pulse_duration <= 0.05:
            valid_peaks.append(peak)
    
    peaks = np.array(valid_peaks)
    peak_times = peaks / sample_rate
    peak_amps = envelope[peaks]
    
    # Filter by reasonable tick count
    if len(peaks) < 8 or len(peaks) > 30:
        assessment = 'OUTLIER'
    else:
        # Calculate pattern metrics
        intervals = np.diff(peak_times)
        
        if len(intervals) > 0:
            total_duration = peak_times[-1] - peak_times[0]
            mean_interval = np.mean(intervals)
            interval_cv = np.std(intervals) / mean_interval if mean_interval > 0 else 999
            amp_cv = np.std(peak_amps) / np.mean(peak_amps) if len(peak_amps) > 0 else 999
            
            # Simple automation criteria
            if len(peaks) >= 12 and interval_cv < 0.6 and total_duration < 6.0:
                assessment = 'AUTOMATED'
            elif len(peaks) >= 10 and interval_cv < 0.8:
                assessment = 'LIKELY_AUTOMATED'
            else:
                assessment = 'MANUAL'
        else:
            assessment = 'INSUFFICIENT_DATA'
            total_duration = mean_interval = interval_cv = amp_cv = 0
    
    return {
        'filename': os.path.basename(wav_file),
        'ticks_found': len(peaks),
        'total_duration': round(total_duration, 2) if 'total_duration' in locals() else 0,
        'mean_interval': round(mean_interval, 3) if 'mean_interval' in locals() else 0,
        'interval_cv': round(interval_cv, 3) if 'interval_cv' in locals() else 0,
        'amplitude_cv': round(amp_cv, 3) if 'amp_cv' in locals() else 0,
        'assessment': assessment,
        'error': None
    }

def analyze_directory(directory_path, output_csv='results.csv'):
    """Analyze all WAV files in directory"""
    wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {directory_path}")
        return None
    
    results = []
    for i, wav_file in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] {os.path.basename(wav_file)}")
        try:
            result = detect_ticks(wav_file)
            results.append(result)
        except Exception as e:
            results.append({
                'filename': os.path.basename(wav_file),
                'ticks_found': 0, 'total_duration': 0, 'mean_interval': 0,
                'interval_cv': 0, 'amplitude_cv': 0, 'assessment': 'ERROR', 'error': str(e)
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Summary
    print(f"\nProcessed {len(results)} files -> {output_csv}")
    print(f"AUTOMATED: {len(df[df['assessment'] == 'AUTOMATED'])}")
    print(f"LIKELY_AUTOMATED: {len(df[df['assessment'] == 'LIKELY_AUTOMATED'])}")
    print(f"MANUAL: {len(df[df['assessment'] == 'MANUAL'])}")
    print(f"ERRORS: {len(df[df['assessment'] == 'ERROR'])}")
    
    return df

# Usage:
# Default 19-26 second window:
# df = analyze_directory('/path/to/wav/files')

# Custom time window:
# df = analyze_directory('/path/to/wav/files', start_time=18, end_time=25)

# Single file with custom window:
# result = detect_ticks('file.wav', start_time=20, end_time=28)
