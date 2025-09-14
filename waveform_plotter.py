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

def plot_waveform(wav_file, start_time=19, end_time=26, show_filtered=True, show_peaks=True, show_spectral=True):
    """
    Plot waveform with tick detection overlay and dead air spectral analysis
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
    num_plots = 2
    if show_filtered: num_plots += 1
    if show_spectral: num_plots += 2  # Spectrogram + dead air analysis
    
    fig_height = 4 * num_plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, fig_height))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Original waveform
    axes[plot_idx].plot(time_axis, audio_segment, color='blue', linewidth=0.8, alpha=0.8)
    axes[plot_idx].set_title(f'Original Audio: {wav_file} ({start_time}-{end_time}s)', fontweight='bold')
    axes[plot_idx].set_ylabel('Amplitude')
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_xlim(start_time, end_time)
    plot_idx += 1
    
    # Spectral analysis
    if show_spectral:
        # Spectrogram
        f, t, Sxx = signal.spectrogram(audio_segment, sample_rate, nperseg=512, noverlap=256)
        t_offset = t + start_time
        
        im = axes[plot_idx].pcolormesh(t_offset, f, 10 * np.log10(Sxx + 1e-10), 
                                      shading='gouraud', cmap='viridis')
        axes[plot_idx].set_title('Spectrogram', fontweight='bold')
        axes[plot_idx].set_ylabel('Frequency (Hz)')
        axes[plot_idx].set_ylim(0, 4000)
        plt.colorbar(im, ax=axes[plot_idx], label='Power (dB)')
        plot_idx += 1
        
        # DEAD AIR FFT ANALYSIS
        print("=== DEAD AIR SPECTRAL ANALYSIS ===")
        
        # Find dead air in extended window (±10 seconds)
        extended_start = max(0, start_sample - int(10.0 * sample_rate))
        extended_end = min(len(audio), end_sample + int(10.0 * sample_rate))
        extended_audio = audio[extended_start:extended_end]
        
        # Dead air detection
        dead_air_threshold = np.percentile(np.abs(extended_audio), 5)
        min_dead_air_duration = int(0.5 * sample_rate)  # 0.5 seconds minimum
        
        print(f"Dead air threshold: {dead_air_threshold:.6f}")
        print(f"Minimum duration: {min_dead_air_duration/sample_rate:.1f} seconds")
        
        # Find quiet segments
        quiet_mask = np.abs(extended_audio) < dead_air_threshold
        quiet_segments = []
        in_quiet = False
        quiet_start = 0
        
        for i, is_quiet in enumerate(quiet_mask):
            if is_quiet and not in_quiet:
                quiet_start = i
                in_quiet = True
            elif not is_quiet and in_quiet:
                if (i - quiet_start) >= min_dead_air_duration:
                    quiet_segments.append((quiet_start, i))
                in_quiet = False
        
        if in_quiet and (len(quiet_mask) - quiet_start) >= min_dead_air_duration:
            quiet_segments.append((quiet_start, len(quiet_mask)))
        
        print(f"Found {len(quiet_segments)} dead air segments")
        
        # FFT ANALYSIS OF DEAD AIR SEGMENTS
        colors = ['red', 'orange', 'purple', 'brown']
        
        for i, (start_idx, end_idx) in enumerate(quiet_segments[:4]):
            segment = extended_audio[start_idx:end_idx]
            segment_duration = len(segment) / sample_rate
            
            print(f"\nSegment {i+1}: {segment_duration:.2f} seconds")
            
            if segment_duration >= 0.5:
                # FFT CALCULATION - THIS IS THE FFT CODE
                freqs = np.fft.rfftfreq(len(segment), 1/sample_rate)
                fft_result = np.fft.rfft(segment)
                power_spectrum = 20 * np.log10(np.abs(fft_result) + 1e-10)
                
                # Plot FFT result
                color = colors[i % len(colors)]
                axes[plot_idx].plot(freqs, power_spectrum, color=color, linewidth=1.5,
                                  label=f'Dead air #{i+1} ({segment_duration:.1f}s)')
                
                # Calculate noise metrics
                freq_60hz_mask = (freqs >= 58) & (freqs <= 62)
                freq_120hz_mask = (freqs >= 118) & (freqs <= 122)
                freq_low_mask = (freqs >= 50) & (freqs <= 200)
                freq_high_mask = (freqs >= 2000) & (freqs <= 4000)
                
                noise_60hz = np.mean(power_spectrum[freq_60hz_mask]) if np.any(freq_60hz_mask) else -999
                noise_120hz = np.mean(power_spectrum[freq_120hz_mask]) if np.any(freq_120hz_mask) else -999
                noise_low = np.mean(power_spectrum[freq_low_mask]) if np.any(freq_low_mask) else -999
                noise_high = np.mean(power_spectrum[freq_high_mask]) if np.any(freq_high_mask) else -999
                
                rms_level = np.sqrt(np.mean(segment**2))
                
                print(f"  RMS: {rms_level:.6f}")
                print(f"  60Hz: {noise_60hz:.1f} dB")
                print(f"  120Hz: {noise_120hz:.1f} dB") 
                print(f"  Low freq: {noise_low:.1f} dB")
                print(f"  High freq: {noise_high:.1f} dB")
                
                # Automation indicators
                if i == 0:
                    print(f"  AUTOMATION INDICATORS:")
                    print(f"    Low RMS (<0.001): {'YES' if rms_level < 0.001 else 'NO'}")
                    print(f"    Clean 60Hz (<-40dB): {'YES' if noise_60hz < -40 else 'NO'}")
                    print(f"    Flat spectrum: {'YES' if abs(noise_high - noise_low) < 10 else 'NO'}")
        
        # Set up the dead air plot
        axes[plot_idx].set_title('Dead Air FFT Spectral Analysis', fontweight='bold')
        axes[plot_idx].set_xlabel('Frequency (Hz)')
        axes[plot_idx].set_ylabel('Power (dB)')
        axes[plot_idx].set_xlim(0, 2000)
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
    
    # Filtered signal analysis
    if show_filtered or show_peaks:
        nyquist = sample_rate / 2
        b, a = signal.butter(4, [1100/nyquist, min(1500/nyquist, 0.99)], btype='band')
        filtered = signal.filtfilt(b, a, audio_segment)
        
        if show_filtered:
            axes[plot_idx].plot(time_axis, filtered, color='green', linewidth=0.8)
            axes[plot_idx].set_title('Bandpass Filtered (1100-1500 Hz)', fontweight='bold')
            axes[plot_idx].set_ylabel('Amplitude')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlim(start_time, end_time)
            plot_idx += 1
        
        if show_peaks:
            # Peak detection with updated parameters
            envelope = np.abs(signal.hilbert(filtered))
            threshold = np.percentile(envelope, 85) * 1.2
            min_peak_distance = int(0.08 * sample_rate)
            
            peaks, _ = signal.find_peaks(envelope, height=threshold, distance=min_peak_distance)
            peak_times = peaks / sample_rate + start_time
            
            # Duration filtering
            valid_peaks = []
            for i, peak in enumerate(peaks):
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
                if 0.005 <= pulse_duration <= 0.08:
                    valid_peaks.append(i)
            
            valid_peak_times = peak_times[valid_peaks]
            
            # Plot peaks
            axes[plot_idx].plot(time_axis, envelope, color='orange', linewidth=1, label='Envelope')
            axes[plot_idx].axhline(threshold, color='red', linestyle='--', linewidth=2, 
                                 label=f'Threshold ({threshold:.4f})')
            
            if len(peaks) > 0:
                axes[plot_idx].scatter(peak_times, envelope[peaks], color='red', s=50, 
                                     marker='x', label=f'All peaks ({len(peaks)})')
            
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
            
            print(f"\n=== TICK DETECTION RESULTS ===")
            print(f"All peaks: {len(peaks)}")
            print(f"Valid ticks: {len(valid_peaks)}")
            if len(valid_peaks) > 1:
                intervals = np.diff(valid_peak_times)
                print(f"Intervals: {[round(x, 3) for x in intervals]}")
                print(f"Mean interval: {np.mean(intervals):.3f}s")
    
    plt.tight_layout()
    plt.show()

def plot_multiple_files(wav_files, start_time=19, end_time=26):
    """Plot multiple files for comparison"""
    for wav_file in wav_files:
        print(f"\n{'='*60}")
        plot_waveform(wav_file, start_time, end_time)

# Usage:
# plot_waveform('call.wav')  # Full analysis with dead air FFT
# plot_waveform('call.wav', show_spectral=False)  # Skip spectral analysis
