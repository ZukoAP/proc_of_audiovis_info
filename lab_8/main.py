# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
# for opening the media file
import librosa
import soundfile as sf
from scipy.signal import argrelextrema


def twiddle(data):
    """Generate the twiddle factors used in the computation of the fast Fourier transform (FFT)
    Args:
        data (int): Number of samples
    Returns:
        sigma (np.ndarray): The twiddle factors
    """
    k = np.arange(data // 2)
    sigma = np.exp(-2j * np.pi * k / data)
    return sigma


def fft(data):  # Cooley–Tukey FFT algorithm radix-2 decimation-in-time
    """Compute the fast Fourier transform (FFT)
    Args:
        data (np.ndarray): Signal to be transformed
    Returns:
        transformed_data (np.ndarray): Fourier transform of x
    """
    data = data.astype(np.complex128)
    data_length = len(data)
    print(data_length)
    log2data = np.log2(data_length)
    assert log2data == int(log2data), 'data_length must be a power of two!'
    transformed_data = np.zeros(data_length, dtype=np.complex128)

    if data_length == 1:
        return data
    else:
        this_range = np.arange(data_length)
        evens = fft(data[this_range % 2 == 0])  # (A(0),…,A(N/2−1)) = DFT_{N/2}*(x(0),x(2),x(4),…,x(N−2))
        odds = fft(data[this_range % 2 == 1])  # (B(0),…,B(N/2−1)) = DFT_{N/2}*(x(1),x(3),x(5),…,x(N−1))
        C = twiddle(data_length) * odds
        transformed_data[:data_length // 2] = evens + C
        transformed_data[data_length // 2:] = evens - C
        return transformed_data


def stft(audio_data, window_function, hop_size=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)
    Args:
        audio_data (np.ndarray): Signal to be transformed
        window_function (np.ndarray): Window function
        hop_size (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)
    Returns:
        result (np.ndarray): The discrete short-time Fourier transform
    """
    window_size = len(window_function)
    data_length = len(audio_data)
    total_segments = np.floor((data_length - window_size) / hop_size).astype(int) + 1
    result = np.zeros((window_size, total_segments), dtype='complex')
    for m in range(total_segments):
        x_win = audio_data[m * hop_size:m * hop_size + window_size] * window_function
        x_win_transformed = np.fft.fft(x_win)
        result[:, m] = x_win_transformed

    if only_positive_frequencies:
        K = 1 + window_size // 2
        result = result[0:K, :]
    return result


def istft(transformed, window_function, hop_size, length):
    """Compute the inverse of the basic discrete short-time Fourier transform (ISTFT)
    Args:
        transformed (np.ndarray): The discrete short-time Fourier transform
        window_function (np.ndarray): Window function
        hop_size (int): Hopsize
        length (int): Length of time signal

    Returns:
        x (np.ndarray): Time signal
    """
    window_size = len(window_function)
    M = transformed.shape[1]
    x_win_sum = np.zeros(length)
    w_sum = np.zeros(length)
    for m in range(M):
        x_win = np.fft.ifft(transformed[:, m])
        # Avoid imaginary values (due to floating point arithmetic)
        x_win = np.real(x_win)
        x_win_sum[m * hop_size:m * hop_size + window_size] = x_win_sum[m * hop_size:m * hop_size + window_size] + x_win
        w_shifted = np.zeros(length)
        w_shifted[m * hop_size:m * hop_size + window_size] = window_function
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum / w_sum
    return x_rec, x_win_sum, w_sum


def noise_estimation_snr(transformed, hop_size):
    """Adaptive noise estimation algorithm.
Estimates the power spectrum of the noise for each frame
    Args:
        transformed (np.array): The discrete short-time Fourier transform
        hop_size (int): Number of frames to use for estimating a-posteriori SNR
    Returns:
        est_Mn, est_Pn (np.array, np.array): magnitude and power spectrum of the noise for each frame
    """
    # Prepare the output variables
    est_Mn = np.zeros(transformed.shape)
    est_Pn = np.zeros(transformed.shape)

    # Iterate through each frame and estimate noise
    for m in range(transformed.shape[0]):
        if m < hop_size:
            # Use noisy spectra for first 10 iterations
            est_Mn[m] = abs(transformed[m])
            est_Pn[m] = est_Mn[m] ** 2
        else:
            a = 25
            # A-posteriori SNR
            gammak = (abs(transformed[m]) ** 2) / np.mean(abs(transformed[m - hop_size:m]) ** 2, axis=0)
            alpha = 1 / (1 + np.exp(-a * (gammak - 1.5)))
            est_Mn[m] = alpha * abs(est_Mn[m - 1]) + (1 - alpha) * abs(transformed[m])
            est_Pn[m] = alpha * (abs(est_Mn[m - 1]) ** 2) + (1 - alpha) * (abs(transformed[m]) ** 2)

    return est_Mn, est_Pn


def spec_subtract_pow(Y, est_Pn):
    est_powX = np.maximum(abs(Y) ** 2 - est_Pn, 0)
    est_phaseX = np.angle(Y)
    est_Sx = np.sqrt(est_powX) * np.exp(1j * est_phaseX)
    return est_Sx


def plot_spectrogram(aud, magnitude, window_size, sample_rate, hop_size):
    fig = plt.figure(figsize=(9, 5))
    gs = grs.GridSpec(1, 2, width_ratios=[100, 1])
    ax1, ax2 = [plt.subplot(gs[i]) for i in range(2)]

    spectrogram = np.abs(magnitude) ** 2
    eps = np.finfo(float).eps
    spec_db = 10 * np.log10(spectrogram + eps)

    time_coefs = np.arange(magnitude.shape[1]) * hop_size / sample_rate
    freq_coefs = np.arange(magnitude.shape[0]) * sample_rate / window_size

    t = np.arange(len(aud)) / sample_rate
    ax1.plot(t, aud, c='gray')
    ax1.set_xlim([min(t), max(t)])
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_xticks(np.arange(min(t), max(t), 0.5))

    ax2.set_visible(False)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(9, 5))
    left = min(time_coefs)
    right = max(time_coefs) + window_size / sample_rate
    lower = min(freq_coefs)
    upper = max(freq_coefs)

    ax3, ax4 = [plt.subplot(gs[i]) for i in range(2)]
    im1 = ax3.imshow(spectrogram, origin='lower', aspect='auto', cmap='gray_r',
                     extent=[left, right, lower, upper])
    # im1.set_clim([np.amin(spectrogram), np.amax(spectrogram)])
    im1.set_clim([0, 10000])
    ax3.set_ylim([lower, upper])
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel("Time (seconds)")
    cbar = fig.colorbar(im1, cax=ax4)
    ax4.set_ylabel('Magnitude (linear)', rotation=90)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(9, 5))
    ax5, ax6 = [plt.subplot(gs[i]) for i in range(2)]

    im2 = ax5.imshow(spec_db, origin='lower', aspect='auto', cmap='gray_r',
                     extent=[left, right, lower, upper])
    # im2.set_clim([np.amin(spec_db), np.amax(spec_db)])
    im2.set_clim([-40, 40])
    ax5.set_ylim([lower, upper])
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(im2, cax=ax6)
    ax6.set_ylabel('Magnitude (dB)', rotation=90)
    plt.tight_layout()
    plt.show()


def plot_local_energy(aud, window_function):
    sqvr = aud ** 2
    energy_local = np.convolve(sqvr, window_function**2, 'same')
    fig = plt.figure(figsize=(9, 5))
    ax1 = plt.subplot()
    t = np.arange(len(aud)) / sample_rate
    ax1.plot(t, energy_local, c='gray')
    ax1.set_xlim([min(t), max(t)])
    ax1.set_ylabel("energy")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_xticks(np.arange(min(t), max(t), 0.5))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wav_file_path = 'guitar.wav'
    aud, sample_rate = librosa.load(wav_file_path)
    # select left channel only
    #  aud = aud[:, 0]
    overlap_factor = 0.75
    window = 2048
    hop_size = np.int32(np.floor(window * (1 - overlap_factor)))
    # our half cosine window
    window_func = np.hanning(window)
    # magnitude = stft(aud, window_func, hop_size)
    magnitude = librosa.stft(aud, n_fft=window, hop_length=hop_size, win_length=window, window='hann', pad_mode='constant', center=False)
    plot_spectrogram(aud, magnitude, window, sample_rate, hop_size)
    est_Mn, est_Pn = noise_estimation_snr(magnitude, 20)
    est_Sx_pow = spec_subtract_pow(magnitude, est_Pn)
    #aud_denoised, _, _ = istft(est_Sx_pow, window_func, hop_size, len(aud))
    aud_denoised = librosa.istft(est_Sx_pow, hop_length=hop_size, win_length=window, window='hann', center=False, length = len(aud))
    min_aud = np.amin(aud)
    max_aud = np.amax(aud)
    aud_denoised = [x for x in aud_denoised if min_aud < x < max_aud]
    plot_spectrogram(aud_denoised, est_Sx_pow, window, sample_rate, hop_size)
    sf.write(wav_file_path[:wav_file_path.find(".wav")]+'_denoised.wav', aud_denoised, sample_rate, subtype='PCM_24')
    plot_local_energy(aud, window_func)
