import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal
import audioop
import os
import pandas as pd
import glob
from datetime import datetime
import argparse

class IVRDialingAnalyzer:
    def __init__(self):
        self.min_ticks = 4
        self.max_ticks = 20
        self.analysis_window = 15.0  # seconds
        
    def read_mulaw_wav(self, wav_file):
        """Read mu-law WAV file"""
        try:
            sample_rate, audio = wavfile.read(wav_file)
            if audio.dtype == np.int16:
                return sample_rate, audio.astype(np.float32) / 32768.0
        except:
            # Mu-law conversion fallback
            try:
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
            except:
                pass
        raise Exception("Could not read file")

    def detect_ticks(self, audio, sample_rate, start_time=0, end_time=None):
        """Detect tick patterns in audio segment"""
        if end_time is None:
            end_time = min(self.analysis_window, len(audio) / sample_rate)
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        if start_sample >= len(audio) or end_sample <= start_sample:
            return [], []
        
        audio_segment = audio[start_sample:min(end_sample, len(audio))]
        
        # Bandpass filter for tick detection (1100-1500 Hz)
        nyquist = sample_rate / 2
        try:
            b, a = signal.butter(4, [1100/nyquist, min(1500/nyquist, 0.99)], btype='band')
            filtered = signal.filtfilt(b, a, audio_segment)
        except:
            return [], []
        
        # Envelope detection
        envelope = np.abs(signal.hilbert(filtered))
        
        # Dynamic threshold based on envelope statistics
        threshold = np.percentile(envelope, 85) * 1.2
        min_peak_distance = int(0.08 * sample_rate)  # 80ms minimum between peaks
        
        # Find peaks
        peaks, _ = signal.find_peaks(envelope, height=threshold, distance=min_peak_distance)
        
        if len(peaks) == 0:
            return [], []
        
        # Filter peaks by duration (valid tick characteristics)
        valid_peaks = []
        for peak in peaks:
            pulse_start = peak
            pulse_end = peak
            
            # Find pulse boundaries
            for j in range(peak, max(0, peak - int(0.1*sample_rate)), -1):
                if envelope[j] < threshold * 0.5:
                    pulse_start = j
                    break
            
            for j in range(peak, min(len(envelope), peak + int(0.1*sample_rate))):
                if envelope[j] < threshold * 0.5:
                    pulse_end = j
                    break
            
            pulse_duration = (pulse_end - pulse_start) / sample_rate
            if 0.005 <= pulse_duration <= 0.08:  # Valid tick duration
                valid_peaks.append(peak)
        
        # Convert to absolute times
        peak_times = np.array(valid_peaks) / sample_rate + start_time
        peak_amplitudes = envelope[valid_peaks]
        
        return peak_times, peak_amplitudes

    def find_dialing_window(self, audio, sample_rate):
        """Find the time window with the highest concentration of ticks"""
        window_size = 8.0  # 8-second sliding window
        step_size = 1.0    # 1-second steps
        
        best_window = (0, self.analysis_window)
        max_ticks = 0
        
        current_time = 0
        while current_time + window_size <= self.analysis_window:
            tick_times, _ = self.detect_ticks(audio, sample_rate, current_time, current_time + window_size)
            
            if len(tick_times) > max_ticks:
                max_ticks = len(tick_times)
                best_window = (current_time, current_time + window_size)
            
            current_time += step_size
        
        return best_window

    def analyze_dialing_pattern(self, tick_times, tick_amplitudes):
        """Analyze tick pattern for automation indicators"""
        if len(tick_times) < self.min_ticks:
            return None
        
        intervals = np.diff(tick_times)
        
        metrics = {
            'tick_count': len(tick_times),
            'duration': tick_times[-1] - tick_times[0],
            'mean_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'cv_timing': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 999,
            'mean_amplitude': np.mean(tick_amplitudes),
            'std_amplitude': np.std(tick_amplitudes),
            'cv_amplitude': np.std(tick_amplitudes) / np.mean(tick_amplitudes) if np.mean(tick_amplitudes) > 0 else 999,
        }
        
        # Advanced timing analysis
        if len(intervals) > 0:
            # Count hesitations (pauses > 2x mean interval)
            long_pauses = np.sum(intervals > 2 * metrics['mean_interval'])
            metrics['hesitation_count'] = long_pauses
            metrics['max_interval'] = np.max(intervals)
            metrics['min_interval'] = np.min(intervals)
        else:
            metrics['hesitation_count'] = 0
            metrics['max_interval'] = 0
            metrics['min_interval'] = 0
        
        # Burst pattern analysis
        if len(intervals) >= 3:
            # Look for natural grouping (account number entry patterns)
            pause_threshold = np.percentile(intervals, 75) * 1.5
            long_pauses_mask = intervals > pause_threshold
            metrics['natural_breaks'] = np.sum(long_pauses_mask)
        else:
            metrics['natural_breaks'] = 0
        
        return metrics

    def classify_dialing(self, metrics):
        """Classify dialing pattern as automated, manual, or uncertain"""
        if metrics is None:
            return "insufficient_data", 0, "Not enough ticks detected"
        
        automation_score = 0
        reasons = []
        
        # Timing regularity (strongest indicator)
        cv_timing = metrics['cv_timing']
        if cv_timing < 0.08:
            automation_score += 40
            reasons.append(f"Very regular timing (CV={cv_timing:.3f})")
        elif cv_timing < 0.15:
            automation_score += 20
            reasons.append(f"Regular timing (CV={cv_timing:.3f})")
        elif cv_timing > 0.3:
            automation_score -= 25
            reasons.append(f"Irregular timing (CV={cv_timing:.3f})")
        
        # Amplitude consistency
        cv_amplitude = metrics['cv_amplitude']
        if cv_amplitude < 0.15:
            automation_score += 25
            reasons.append(f"Consistent amplitude (CV={cv_amplitude:.3f})")
        elif cv_amplitude > 0.4:
            automation_score -= 15
            reasons.append(f"Variable amplitude (CV={cv_amplitude:.3f})")
        
        # Speed analysis
        mean_interval = metrics['mean_interval']
        if 0.08 <= mean_interval <= 0.15 and cv_timing < 0.1:
            automation_score += 30
            reasons.append(f"Machine-like speed ({mean_interval:.3f}s intervals)")
        elif mean_interval < 0.08:
            automation_score += 35
            reasons.append(f"Too fast for human ({mean_interval:.3f}s intervals)")
        
        # Human indicators
        if metrics['hesitation_count'] >= 2:
            automation_score -= 20
            reasons.append(f"Multiple hesitations ({metrics['hesitation_count']})")
        
        if metrics['natural_breaks'] >= 2:
            automation_score -= 15
            reasons.append(f"Natural grouping pattern ({metrics['natural_breaks']} breaks)")
        
        # Very long sequences suggest complete account entry
        if metrics['tick_count'] >= 14:
            automation_score += 10
            reasons.append(f"Complete entry ({metrics['tick_count']} digits)")
        
        # Final classification
        if automation_score >= 50:
            classification = "AUTOMATED"
        elif automation_score <= 15:
            classification = "MANUAL"
        else:
            classification = "UNCERTAIN"
        
        confidence = min(abs(automation_score - 32.5) / 32.5 * 100, 100)
        
        return classification, confidence, "; ".join(reasons)

    def analyze_file(self, wav_file):
        """Analyze a single WAV file"""
        result = {
            'filename': os.path.basename(wav_file),
            'status': 'error',
            'classification': None,
            'confidence': 0,
            'reasons': '',
            'tick_count': 0,
            'analysis_window': None
        }
        
        try:
            # Read audio
            sample_rate, audio = self.read_mulaw_wav(wav_file)
            
            # Check minimum duration
            duration = len(audio) / sample_rate
            if duration < 3.0:
                result['status'] = 'too_short'
                result['reasons'] = f'File too short ({duration:.1f}s)'
                return result
            
            # Find optimal analysis window
            window_start, window_end = self.find_dialing_window(audio, sample_rate)
            result['analysis_window'] = f"{window_start:.1f}-{window_end:.1f}s"
            
            # Detect ticks in optimal window
            tick_times, tick_amplitudes = self.detect_ticks(audio, sample_rate, window_start, window_end)
            
            if len(tick_times) < self.min_ticks:
                result['status'] = 'no_ticks'
                result['tick_count'] = len(tick_times)
                result['reasons'] = f'Insufficient ticks ({len(tick_times)} found, need {self.min_ticks})'
                return result
            
            # Analyze pattern
            metrics = self.analyze_dialing_pattern(tick_times, tick_amplitudes)
            classification, confidence, reasons = self.classify_dialing(metrics)
            
            result.update({
                'status': 'analyzed',
                'classification': classification,
                'confidence': round(confidence, 1),
                'reasons': reasons,
                'tick_count': len(tick_times),
                'mean_interval': round(metrics['mean_interval'], 3),
                'cv_timing': round(metrics['cv_timing'], 3),
                'cv_amplitude': round(metrics['cv_amplitude'], 3),
                'hesitation_count': metrics['hesitation_count']
            })
            
        except Exception as e:
            result['reasons'] = f'Error: {str(e)}'
        
        return result

    def analyze_directory(self, directory_path, output_file=None):
        """Analyze all WAV files in directory"""
        wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
        
        if not wav_files:
            print(f"No WAV files found in {directory_path}")
            return None
        
        print(f"Found {len(wav_files)} WAV files to analyze...")
        
        results = []
        for i, wav_file in enumerate(wav_files, 1):
            if i % 50 == 0 or i == len(wav_files):
                print(f"Processing file {i}/{len(wav_files)}")
            
            result = self.analyze_file(wav_file)
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate summary
        self.print_summary(df)
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ivr_analysis_results_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return df

    def print_summary(self, df):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("IVR DIALING PATTERN ANALYSIS SUMMARY")
        print("="*60)
        
        total_files = len(df)
        analyzed_files = len(df[df['status'] == 'analyzed'])
        
        print(f"Total files processed: {total_files}")
        print(f"Successfully analyzed: {analyzed_files}")
        print(f"Too short: {len(df[df['status'] == 'too_short'])}")
        print(f"No ticks detected: {len(df[df['status'] == 'no_ticks'])}")
        print(f"Errors: {len(df[df['status'] == 'error'])}")
        
        if analyzed_files > 0:
            analyzed_df = df[df['status'] == 'analyzed']
            
            print(f"\nCLASSIFICATION RESULTS:")
            print(f"AUTOMATED: {len(analyzed_df[analyzed_df['classification'] == 'AUTOMATED'])}")
            print(f"MANUAL: {len(analyzed_df[analyzed_df['classification'] == 'MANUAL'])}")
            print(f"UNCERTAIN: {len(analyzed_df[analyzed_df['classification'] == 'UNCERTAIN'])}")
            
            # High confidence results
            high_conf = analyzed_df[analyzed_df['confidence'] >= 70]
            print(f"\nHigh confidence (≥70%): {len(high_conf)} files")
            if len(high_conf) > 0:
                auto_high = len(high_conf[high_conf['classification'] == 'AUTOMATED'])
                manual_high = len(high_conf[high_conf['classification'] == 'MANUAL'])
                print(f"  AUTOMATED: {auto_high}")
                print(f"  MANUAL: {manual_high}")
            
            print(f"\nTICK COUNT STATISTICS:")
            print(f"Mean ticks per file: {analyzed_df['tick_count'].mean():.1f}")
            print(f"Range: {analyzed_df['tick_count'].min()}-{analyzed_df['tick_count'].max()}")
            
            # Files with full account numbers (likely 16 digits)
            full_entries = analyzed_df[analyzed_df['tick_count'] >= 14]
            if len(full_entries) > 0:
                print(f"\nFULL ACCOUNT ENTRIES (≥14 ticks): {len(full_entries)}")
                auto_full = len(full_entries[full_entries['classification'] == 'AUTOMATED'])
                print(f"  AUTOMATED: {auto_full}/{len(full_entries)} ({100*auto_full/len(full_entries):.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze IVR recordings for automated vs manual dialing')
    parser.add_argument('directory', help='Directory containing WAV files')
    parser.add_argument('-o', '--output', help='Output CSV filename')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' not found")
        return
    
    analyzer = IVRDialingAnalyzer()
    analyzer.analyze_directory(args.directory, args.output)

if __name__ == "__main__":
    main()
