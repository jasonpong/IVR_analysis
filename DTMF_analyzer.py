import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram, butter, filtfilt
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
from IPython.display import display, Audio
import pandas as pd

class DTMFAnalyzerJupyter:
    """DTMF Analyzer optimized for Jupyter notebooks with visualization"""
    
    # DTMF frequency pairs (low freq, high freq) in Hz
    DTMF_FREQS = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
    }
    
    # DTMF frequencies for detection
    LOW_FREQS = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477, 1633]
    
    def __init__(self, wav_file: str, threshold_db: float = -30):
        """Initialize DTMF analyzer"""
        self.wav_file = wav_file
        self.threshold_db = threshold_db
        self.sample_rate, self.audio_data = self._load_wav()
        self.tones = None
        self.intervals = None
        
    def _load_wav(self) -> Tuple[int, np.ndarray]:
        """Load WAV file and convert to mono if needed"""
        sample_rate, audio = wavfile.read(self.wav_file)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize to [-1, 1]
        if audio.dtype == np.int16:
            audio = audio / 32768.0
        elif audio.dtype == np.int32:
            audio = audio / 2147483648.0
            
        return sample_rate, audio
    
    def goertzel_algorithm(self, samples: np.ndarray, freq: float) -> float:
        """Goertzel algorithm for efficient single-frequency DFT"""
        N = len(samples)
        k = int(0.5 + N * freq / self.sample_rate)
        omega = 2 * np.pi * k / N
        coeff = 2 * np.cos(omega)
        
        s_prev = 0
        s_prev2 = 0
        
        for sample in samples:
            s = sample + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
            
        power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
        return np.sqrt(power) / N
    
    def detect_dtmf_tones(self, window_ms: int = 40, step_ms: int = 20) -> List[Dict]:
        """Detect DTMF tones in the audio"""
        window_size = int(self.sample_rate * window_ms / 1000)
        step_size = int(self.sample_rate * step_ms / 1000)
        
        detected_tones = []
        current_tone = None
        tone_start = None
        
        # Sliding window analysis
        for i in range(0, len(self.audio_data) - window_size, step_size):
            window = self.audio_data[i:i + window_size]
            
            # Apply window function to reduce spectral leakage
            window = window * np.hanning(len(window))
            
            # Calculate energy for each DTMF frequency
            low_energies = [self.goertzel_algorithm(window, f) for f in self.LOW_FREQS]
            high_energies = [self.goertzel_algorithm(window, f) for f in self.HIGH_FREQS]
            
            # Find peak frequencies
            low_idx = np.argmax(low_energies)
            high_idx = np.argmax(high_energies)
            
            # Convert to dB
            low_energy_db = 20 * np.log10(low_energies[low_idx] + 1e-10)
            high_energy_db = 20 * np.log10(high_energies[high_idx] + 1e-10)
            
            # Check if both frequencies exceed threshold
            if low_energy_db > self.threshold_db and high_energy_db > self.threshold_db:
                detected_freq_pair = (self.LOW_FREQS[low_idx], self.HIGH_FREQS[high_idx])
                
                # Find corresponding DTMF digit
                digit = self._freq_pair_to_digit(detected_freq_pair)
                
                if digit:
                    current_time = i / self.sample_rate
                    
                    if current_tone != digit:
                        # New tone detected
                        if current_tone and tone_start is not None:
                            # Save previous tone
                            detected_tones.append({
                                'digit': current_tone,
                                'start_time': tone_start,
                                'end_time': current_time,
                                'duration': current_time - tone_start
                            })
                        
                        current_tone = digit
                        tone_start = current_time
            else:
                # No tone detected
                if current_tone and tone_start is not None:
                    current_time = i / self.sample_rate
                    detected_tones.append({
                        'digit': current_tone,
                        'start_time': tone_start,
                        'end_time': current_time,
                        'duration': current_time - tone_start
                    })
                    current_tone = None
                    tone_start = None
        
        # Handle last tone if still active
        if current_tone and tone_start is not None:
            end_time = len(self.audio_data) / self.sample_rate
            detected_tones.append({
                'digit': current_tone,
                'start_time': tone_start,
                'end_time': end_time,
                'duration': end_time - tone_start
            })
        
        self.tones = detected_tones
        return detected_tones
    
    def _freq_pair_to_digit(self, freq_pair: Tuple[float, float]) -> str:
        """Convert frequency pair to DTMF digit"""
        for digit, freqs in self.DTMF_FREQS.items():
            if abs(freqs[0] - freq_pair[0]) < 50 and abs(freqs[1] - freq_pair[1]) < 50:
                return digit
        return None
    
    def calculate_intervals(self, tones: List[Dict]) -> List[Dict]:
        """Calculate intervals between consecutive tones"""
        intervals = []
        
        for i in range(1, len(tones)):
            interval = {
                'from_digit': tones[i-1]['digit'],
                'to_digit': tones[i]['digit'],
                'interval': tones[i]['start_time'] - tones[i-1]['end_time'],
                'from_end': tones[i-1]['end_time'],
                'to_start': tones[i]['start_time']
            }
            intervals.append(interval)
        
        self.intervals = intervals
        return intervals
    
    def generate_square_wave(self) -> np.ndarray:
        """Generate square wave representation of detected tones"""
        if self.tones is None:
            self.detect_dtmf_tones()
        
        # Create time axis matching original audio
        square_wave = np.zeros_like(self.audio_data)
        
        # For each detected tone, create a square pulse
        for tone in self.tones:
            start_sample = int(tone['start_time'] * self.sample_rate)
            end_sample = int(tone['end_time'] * self.sample_rate)
            
            # Different amplitude for each digit (for visual distinction)
            digit_value = int(tone['digit']) if tone['digit'].isdigit() else 10
            amplitude = 0.5 + (digit_value * 0.05)
            
            square_wave[start_sample:end_sample] = amplitude
            
        return square_wave
    
    def plot_comparison(self, figsize=(15, 6)):
        """Create visualization comparing original and detected tones"""
        if self.tones is None:
            self.detect_dtmf_tones()
            
        square_wave = self.generate_square_wave()
        time_axis = np.arange(len(self.audio_data)) / self.sample_rate
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # 1. Original waveform
        axes[0].plot(time_axis, self.audio_data, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].set_title(f'Original Audio Waveform - {os.path.basename(self.wav_file)}', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-1.1, 1.1])
        
        # Add tone labels on original waveform
        for tone in self.tones:
            mid_time = (tone['start_time'] + tone['end_time']) / 2
            axes[0].text(mid_time, 0.8, tone['digit'], 
                        ha='center', va='center', fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 2. Square wave representation
        axes[1].plot(time_axis, square_wave, 'r-', linewidth=1.5)
        axes[1].fill_between(time_axis, 0, square_wave, alpha=0.3, color='red')
        axes[1].set_ylabel('DTMF Active', fontsize=10)
        axes[1].set_xlabel('Time (seconds)', fontsize=10)
        axes[1].set_title('Detected DTMF Tones (Square Wave Representation)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-0.1, 1.1])
        
        # Add digit labels on square wave
        for tone in self.tones:
            mid_time = (tone['start_time'] + tone['end_time']) / 2
            digit_value = int(tone['digit']) if tone['digit'].isdigit() else 10
            amplitude = 0.5 + (digit_value * 0.05)
            axes[1].text(mid_time, amplitude + 0.1, tone['digit'], 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add duration annotation
            axes[1].annotate('', xy=(tone['start_time'], -0.05), xytext=(tone['end_time'], -0.05),
                           arrowprops=dict(arrowstyle='<->', color='green', lw=1))
            axes[1].text((tone['start_time'] + tone['end_time'])/2, -0.08, 
                        f"{tone['duration']*1000:.0f}ms", 
                        ha='center', va='top', fontsize=8, color='green')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_timing_analysis(self, figsize=(12, 5)):
        """Plot timing analysis of tones and intervals"""
        if self.tones is None:
            self.detect_dtmf_tones()
        if self.intervals is None:
            self.calculate_intervals(self.tones)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Tone durations
        if self.tones:
            durations = [t['duration'] * 1000 for t in self.tones]
            digits = [t['digit'] for t in self.tones]
            
            axes[0].bar(range(len(durations)), durations, color='steelblue', alpha=0.7)
            axes[0].set_xticks(range(len(digits)))
            axes[0].set_xticklabels(digits)
            axes[0].set_xlabel('Digit', fontsize=10)
            axes[0].set_ylabel('Duration (ms)', fontsize=10)
            axes[0].set_title('Tone Durations', fontsize=12, fontweight='bold')
            axes[0].axhline(y=np.mean(durations), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(durations):.1f}ms')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Inter-tone intervals
        if self.intervals:
            intervals = [i['interval'] * 1000 for i in self.intervals]
            labels = [f"{i['from_digit']}-{i['to_digit']}" for i in self.intervals]
            
            axes[1].bar(range(len(intervals)), intervals, color='coral', alpha=0.7)
            axes[1].set_xticks(range(len(labels)))
            axes[1].set_xticklabels(labels, rotation=45)
            axes[1].set_xlabel('Transition', fontsize=10)
            axes[1].set_ylabel('Interval (ms)', fontsize=10)
            axes[1].set_title('Inter-tone Intervals', fontsize=12, fontweight='bold')
            axes[1].axhline(y=np.mean(intervals), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(intervals):.1f}ms')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def display_summary(self):
        """Display summary statistics in Jupyter notebook"""
        if self.tones is None:
            self.detect_dtmf_tones()
        if self.intervals is None:
            self.calculate_intervals(self.tones)
        
        # Create summary DataFrame
        print("=" * 60)
        print(f"DTMF Analysis Summary for: {os.path.basename(self.wav_file)}")
        print("=" * 60)
        
        print(f"\nAudio Properties:")
        print(f"  Duration: {len(self.audio_data)/self.sample_rate:.2f} seconds")
        print(f"  Sample Rate: {self.sample_rate} Hz")
        print(f"  Total Tones Detected: {len(self.tones)}")
        
        if self.tones:
            # Tone sequence
            sequence = ' '.join([t['digit'] for t in self.tones])
            print(f"\nDetected Sequence: {sequence}")
            
            # Tones DataFrame
            df_tones = pd.DataFrame(self.tones)
            df_tones['duration_ms'] = df_tones['duration'] * 1000
            df_tones = df_tones[['digit', 'start_time', 'end_time', 'duration_ms']]
            
            print("\nTone Details:")
            display(df_tones.round(3))
            
            # Statistics
            durations = [t['duration'] * 1000 for t in self.tones]
            print(f"\nTone Duration Statistics (ms):")
            print(f"  Mean: {np.mean(durations):.1f}")
            print(f"  Std Dev: {np.std(durations):.1f}")
            print(f"  Min: {np.min(durations):.1f}")
            print(f"  Max: {np.max(durations):.1f}")
            
            if self.intervals:
                interval_times = [i['interval'] * 1000 for i in self.intervals]
                print(f"\nInterval Statistics (ms):")
                print(f"  Mean: {np.mean(interval_times):.1f}")
                print(f"  Std Dev: {np.std(interval_times):.1f}")
                print(f"  Min: {np.min(interval_times):.1f}")
                print(f"  Max: {np.max(interval_times):.1f}")
                
                # Check for automated pattern
                if np.std(durations) < 10 and np.std(interval_times) < 20:
                    print("\n⚠️ AUTOMATED PATTERN DETECTED: Very consistent timing suggests automated input")
                else:
                    print("\n👤 HUMAN PATTERN DETECTED: Variable timing suggests human input")
    
    def play_audio(self):
        """Play the audio file in Jupyter notebook"""
        print(f"Playing: {os.path.basename(self.wav_file)}")
        return Audio(self.audio_data, rate=self.sample_rate)


# Convenience function for quick analysis
def analyze_dtmf(wav_file, threshold_db=-30, show_plots=True):
    """
    Quick DTMF analysis function for Jupyter notebooks
    
    Args:
        wav_file: Path to WAV file
        threshold_db: Detection threshold in dB
        show_plots: Whether to display plots
    
    Returns:
        DTMFAnalyzerJupyter instance
    """
    analyzer = DTMFAnalyzerJupyter(wav_file, threshold_db)
    
    # Detect tones and calculate intervals
    analyzer.detect_dtmf_tones()
    analyzer.calculate_intervals(analyzer.tones)
    
    # Display summary
    analyzer.display_summary()
    
    # Show plots if requested
    if show_plots:
        analyzer.plot_comparison()
        analyzer.plot_timing_analysis()
    
    return analyzer


# Example usage in Jupyter cell
if __name__ == "__main__":
    # This would be in a Jupyter cell:
    # analyzer = analyze_dtmf("test_867_5309.wav")
    # 
    # # Play the audio
    # analyzer.play_audio()
    #
    # # Get the square wave data if needed
    # square_wave = analyzer.generate_square_wave()
    pass
