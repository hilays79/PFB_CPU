import numpy as np
import matplotlib.pyplot as plt

# Parameters
f_signal = 50.0  # Signal frequency in Hz
f_sample = 210.0 # Sampling frequency in Hz
duration = 1 / f_signal # Single period

# Calculate number of samples:
N = int(f_sample * duration)
t = np.arange(N) / f_sample

# Create a sine and cosine signal
y_sin = np.sin(2 * np.pi * f_signal * t)
y_cos = np.cos(2 * np.pi * f_signal * t)

# Compute DFT
fft_sin = np.fft.fft(y_sin)
fft_cos = np.fft.fft(y_cos)

# Compute corresponding frequency bins
freqs = np.fft.fftfreq(N, 1/f_sample)

# 4. Shift the arrays to center 0 Hz
freqs_shifted = np.fft.fftshift(freqs)
mag_sin_shifted = np.fft.fftshift(np.abs(fft_sin))
mag_cos_shifted = np.fft.fftshift(np.abs(fft_cos))

# 5. Plotting
plt.figure(figsize=(12, 5))

# Plot Sine Spectrum
plt.subplot(1, 2, 1)
# Using stem instead of plot because frequency bins are discrete values
plt.stem(freqs_shifted, mag_sin_shifted, basefmt="k-") 
plt.title("Magnitude of Sine DFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(-55, 55) # Restricting the view to +/- 60 Hz
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Cosine Spectrum
plt.subplot(1, 2, 2)
plt.stem(freqs_shifted, mag_cos_shifted, basefmt="k-")
plt.title("Magnitude of Cosine DFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(-55, 55) # Restricting the view to +/- 60 Hz
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()