import constants
import numpy as np
from scipy.signal import cwt, resample, ricker
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_features_from_spectrogram_basic(
    spectrogram: np.ndarray,
    n_time_bins: int = constants.TIME_BINS,
    n_freq_bins: int = constants.FREQUENCY_BINS,
) -> np.ndarray:
    """
    Extracts features from a spectrogram by averaging over time and frequency bins.

    Parameters:
    - spectrogram (np.ndarray): The input spectrogram (frequency x time).
    - n_time_bins (int): The number of time bins to aggregate (fixed for all samples).
    - n_freq_bins (int): The number of frequency bins to aggregate.

    Returns:
    - np.ndarray: The extracted feature vector.
    """
    resampled_spectrogram = resample(spectrogram, n_time_bins, axis=1)

    # Split frequency bins into n_freq_bins zones
    freq_bin_size = resampled_spectrogram.shape[0] // n_freq_bins

    frequency_features = []

    for i in range(n_freq_bins):
        start_bin = i * freq_bin_size
        end_bin = (i + 1) * freq_bin_size
        # Average the magnitude over each frequency zone
        zone_mean = np.mean(resampled_spectrogram[start_bin:end_bin], axis=0)
        frequency_features.append(zone_mean)

    # Flatten frequency features into a single vector
    feature_vector = np.concatenate(frequency_features)
    return feature_vector


def renyi_entropy(TFR, alpha=3, t=None, f=None):
    """
    Oblicza entropię Rényiego dla podanego TFR (Time-Frequency Representation).

    Parametry:
    - TFR: 2D numpy array (freq x time)
    - alpha: rząd entropii Rényiego (domyślnie 3)
    - t: wektor czasu, jeśli None, przyjmujemy indeksy
    - f: wektor częstotliwości, jeśli None, przyjmujemy indeksy

    Zwraca:
    - Ren: skalar, wartość entropii Rényiego
    """
    M, N = TFR.shape
    if t is None:
        t = np.arange(N)
    if f is None:
        f = np.arange(M)

    # Sortowanie dla pewności, choć tutaj raczej zbędne, zakładamy f rosnące
    f = np.sort(f)

    # Normalizacja TFR:
    total_integral = np.trapz(np.trapz(TFR, t, axis=1), f, axis=0)
    TFR = TFR / total_integral

    eps = np.finfo(float).eps

    if alpha == 1:
        # Shannon entropy (limit dla alpha=1)
        # Sprawdź, czy nie ma ujemnych wartości:
        if np.min(TFR) < 0:
            raise ValueError(
                "Distribution with negative values => alpha=1 not allowed."
            )
        Ren = -np.trapz(np.trapz(TFR * np.log2(TFR + eps), t, axis=1), f, axis=0)
    else:
        # Renyi entropy
        val = np.trapz(np.trapz(TFR**alpha, t, axis=1), f, axis=0) + eps
        Ren = 1 / (1 - alpha) * np.log2(val)

    return Ren


def extract_features_from_spectrogram_advanced(
    spectrogram_data,
):
    """
    Extract advanced features from a spectrogram for gesture recognition.

    Parameters:
    - spectrogram_data: 2D numpy array (frequency x time)

    Returns:
    - features: 1D numpy array of extracted features
    """
    # Normalize spectrogram
    normalized_spectrogram_data = (
        spectrogram_data - np.mean(spectrogram_data)
    ) / np.std(spectrogram_data)

    # Compute higher-order statistics
    mean_val = np.mean(normalized_spectrogram_data)
    var_val = np.var(normalized_spectrogram_data)
    skew_val = skew(normalized_spectrogram_data.flatten())
    kurtosis_val = kurtosis(normalized_spectrogram_data.flatten())

    average_over_time_and_frequency_bins = extract_features_from_spectrogram_basic(
        normalized_spectrogram_data
    )

    # Wavelet Transform
    widths = np.arange(1, 31)
    cwt_matr = cwt(normalized_spectrogram_data.flatten(), ricker, widths)
    cwt_mean = np.mean(cwt_matr, axis=1)
    cwt_var = np.var(cwt_matr, axis=1)

    # Entropia Rényiego (alpha=3)
    t = np.arange(spectrogram_data.shape[1])
    f = np.arange(spectrogram_data.shape[0])
    renyi_val = renyi_entropy(np.abs(normalized_spectrogram_data), alpha=3, t=t, f=f)

    # Concatenate all features
    features = np.concatenate(
        [
            [mean_val, var_val, skew_val, kurtosis_val],
            # micro_doppler_features, |deprecated v2 version
            average_over_time_and_frequency_bins,
            cwt_mean,
            cwt_var,
            [renyi_val],
        ]
    )

    return features


def normalize_features(features: np.ndarray, labels: np.ndarray = None):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features, labels)
    return normalized_features


def apply_pca(data, labels, num_components=50):
    pca = PCA(n_components=num_components)
    return pca.fit_transform(data, labels)
