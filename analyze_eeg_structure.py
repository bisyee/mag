import numpy as np
import matplotlib.pyplot as plt
import random, os, mat73, scipy.io
from scipy.signal import butter, filtfilt

class PDBrainProcessor:
    def __init__(self, srate):
        self.srate = srate
        self.bands = {
            'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
            'Beta (Motor)': (13, 30), 'Tremor': (3, 6)
        }

    def filter_signal(self, data, low, high):
        nyq = 0.5 * self.srate
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        # This handles both 2D (chan x time) and 3D (trial x chan x time) automatically
        return filtfilt(b, a, data, axis=-1)

    def plot_analysis(self, raw_data, subject):
        # Prepare 1D signal for plotting (first trial, first channel)
        sig = raw_data[0, 0, :] if raw_data.ndim == 3 else raw_data[0, :]
        t = np.arange(len(sig)) / self.srate
        win = (int(10*self.srate), int(15*self.srate)) # 5s window
        
        fig, axes = plt.subplots(len(self.bands)+1, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(t[win[0]:win[1]], sig[win[0]:win[1]], 'k', lw=0.7)
        axes[0].set_title(f"Original: {subject}")

        for i, (name, (low, high)) in enumerate(self.bands.items(), 1):
            filtered = self.filter_signal(sig, low, high)
            axes[i].plot(t[win[0]:win[1]], filtered[win[0]:win[1]], lw=0.7)
            axes[i].set_title(f"{name} ({low}-{high}Hz)")
            
            # Power Analysis
            pwr = np.sqrt(np.mean(filtered**2))
            print(f"{name} Power: {pwr:.4f}")

        plt.tight_layout()
        plt.show()

def load_eeg(path):
    try:
        data = mat73.loadmat(path) if '.mat' in path else None
    except:
        data = scipy.io.loadmat(path)
    
    eeg = data.get('EEG', {})
    if hasattr(eeg, 'dtype'): # Scipy struct handling
        eeg = {n: eeg[0,0][n] for n in eeg.dtype.names}
    return eeg

def main():
    path = os.path.join(os.getcwd(), 'PD_REST')
    files = [f for f in os.listdir(path) if f.endswith('.mat') and '1' not in f]
    
    for file in random.sample(files, min(2, len(files))):
        info = load_eeg(os.path.join(path, file))
        if info.get('data') is not None:
            print(f"\n--- Analyzing {file} ---")
            proc = PDBrainProcessor(info['srate'])
            proc.plot_analysis(info['data'], file)

if __name__ == "__main__":
    main()