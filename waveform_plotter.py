import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal
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

def plot_waveform(wav_file, start_time=19, end_time=26, show_filtered=True, show_peaks=True):
    """
    Plot waveform with tick detection overlay
    
    Args:
        wav_file: Path to WAV file
        start_time: Start time in seconds
        end_time: End time in seconds
        show_filtered: Show bandpass filtered signal
        show_peaks: Show detected tick peaks
    """
    
    # Read audio
    sample_rate, audio = read_mulaw_wav(wav_file)
    
    # Extract time window
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    if start_sample >= len(audio):
        print("Error: Start time beyond audio duration")
        return
    
    audio_segment = audio[start_sample:min(end_sample, len(audio))]
    time_axis = np.linspace(start_time, start_time + len(audio_segment)/sample_rate, len(audio_segment))
    
    # Set up plot
    fig_height = 12 if show_filtered else 8
    fig, axes = plt.subplots(3 if show_filtered else 2, 1, figsize=(15, fig_height))
    if not show_filtered:
        axes = [axes[0], axes[1], None]
    
    # Plot 1: Original waveform
    axes[0].plot(time_axis, audio_segment, color='blue', linewidth=0.8, alpha=0.8)
    axes[0].set_title(f'Original Audio: {wav_file} ({start_time}-{end_time}s)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(start_time, end_time)
    
    if show_filtered or show_peaks:
        # Apply bandpass filter for analysis
        nyquist = sample_rate / 2
        b, a = signal.butter(4, [1100/nyquist, min(1500/nyquist, 0.99)], btype='band')
        filtered = signal.filtfilt(b, a, audio_segment)
        
        if show_filtered:
            # Plot 2: Filtered signal
            axes[1].plot(time_axis, filtered, color='green', linewidth=0.8)
            axes[1].set_title('Bandpass Filtered (1100-1500 Hz)', fontweight='bold')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(start_time, end_time)
        
        if show_peaks:
            # Peak detection
            envelope = np.abs(signal.hilbert(filtered))
            threshold = np.percentile(envelope, 90) * 1.5
            min_peak_distance = int(0.1 * sample_rate)  # 100ms separation
            
            peaks, _ = signal.find_peaks(envelope, height=threshold, distance=min_peak_distance)
            peak_times = peaks / sample_rate + start_time
            
            # Filter by duration
            valid_peaks = []
            for i, peak in enumerate(peaks):
                # Find pulse boundaries
                pulse_start = peak
                pulse_end = peak
                
                for j in range(peak, max(0, peak - int(0.1*sample_rate)), -1):
                    if envelope[j] < threshold * 0.5:
                        pulse_start = j
                        break
                
                for j in range(peak, min(len(envelope), peak + int(0.1*sample_rate))):
                    if envelope[j] < threshold * 0.5:
                        pulse_end = j
                        break
                
                pulse_duration = (pulse_end - pulse_start) / sample_rate
                if 0.01 <= pulse_duration <= 0.05:  # 10-50ms
                    valid_peaks.append(i)
            
            valid_peak_times = peak_times[valid_peaks]
            
            # Plot envelope with peaks
            plot_idx = 2 if show_filtered else 1
            axes[plot_idx].plot(time_axis, envelope, color='orange', linewidth=1, label='Envelope')
            axes[plot_idx].axhline(threshold, color='red', linestyle='--', linewidth=2, 
                                 label=f'Threshold ({threshold:.4f})')
            
            # Mark all detected peaks
            if len(peaks) > 0:
                axes[plot_idx].scatter(peak_times, envelope[peaks], color='red', s=50, 
                                     marker='x', label=f'All peaks ({len(peaks)})')
            
            # Mark valid tick peaks
            if len(valid_peaks) > 0:
                axes[plot_idx].scatter(valid_peak_times, envelope[peaks[valid_peaks]], 
                                     color='lime', s=100, marker='o', 
                                     label=f'Valid ticks ({len(valid_peaks)})')
            
            axes[plot_idx].set_title('Peak Detection Analysis', fontweight='bold')
            axes[plot_idx].set_xlabel('Time (seconds)')
            axes[plot_idx].set_ylabel('Envelope')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlim(start_time, end_time)
            
            # Print results
            print(f"File: {wav_file}")
            print(f"Time window: {start_time}-{end_time}s")
            print(f"All peaks found: {len(peaks)}")
            print(f"Valid ticks: {len(valid_peaks)}")
            if len(valid_peaks) > 1:
                intervals = np.diff(valid_peak_times)
                print(f"Tick intervals: {[round(x, 3) for x in intervals]}")
                print(f"Mean interval: {np.mean(intervals):.3f}s")
    
    plt.tight_layout()
    plt.show()

def plot_multiple_files(wav_files, start_time=19, end_time=26):
    """Plot multiple files for comparison"""
    for wav_file in wav_files:
        print(f"\n{'='*50}")
        plot_waveform(wav_file, start_time, end_time)

# Usage examples:
# plot_waveform('call.wav')  # Default 19-26 seconds
# plot_waveform('call.wav', 18, 25)  # Custom time range
# plot_waveform('call.wav', show_filtered=False)  # Just original + peaks
# plot_multiple_files(['file1.wav', 'file2.wav'])  # Compare multiple files
