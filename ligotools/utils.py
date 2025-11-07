import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- Utility 1: whiten ---
def whiten(strain, psd_interp, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    white_hf = hf / (np.sqrt(psd_interp(freqs) / (dt / 2.0)))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

# --- Utility 2: write_wavfile ---
def write_wavfile(filename, fs, data):
    """Normalize and write data to a .wav file"""
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

# --- Utility 3: reqshift ---
def reqshift(data, fshift=100, sample_rate=4096):
    """Frequency shift the signal by constant"""
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.0
    z = np.fft.irfft(y)
    return z

# --- Utility 4: plot_psd ---
def plot_psd(freqs, Pxx_L1, Pxx_H1, Pxx, f_min, f_max, eventname, plottype, FIGS):
    """Plot power spectral density for both detectors"""
    plt.figure(figsize=(10, 8))
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'g', label='L1 strain')
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'r', label='H1 strain')
    plt.loglog(freqs, np.sqrt(Pxx), 'k', label='H1 strain, O1 smooth model')
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid(True)
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='upper center')
    plt.title(f'Advanced LIGO strain data near {eventname}')
    plt.savefig(FIGS / f"{eventname}_ASDs.{plottype}")
    plt.close()

def plot_matchfreq(datafreq, template_fft, d_eff, freqs, data_psd, det, eventname, fs, plottype, FIGS):
    """Plot and save the ASD and template for each detector."""
    plt.figure(figsize=(10, 6))
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    pcolor = 'g' if det == 'L1' else 'r'
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=det + ' ASD')
    plt.xlim(20, fs / 2)
    plt.ylim(1e-24, 1e-20)
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det + ' ASD and template around event')
    plt.savefig(FIGS / f"{eventname}_{det}_matchfreq.{plottype}")
    plt.close()

