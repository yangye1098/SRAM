"""
Licensed under GNU General Public License GPLv3
"""

import numpy as np
import scipy.signal as sig
from math import floor, sqrt
from matplotlib.mlab import psd, window_none, window_hanning
import itertools


def octaveBandSpectra(octave_band_env, hz, psd_window, window_type='none', norm='env_amp_square'):
    """
    Calculate octave band power spectra
    Input
    -----
    * octave_band_env : ndarray in [n_bands, N]
        Octave band envelope from octave band audio
    * hz : float or int
        Audio sample rate in hertz
    * psd_window : int
        psd window length in sample points

    Output
    ------
    * spectra : ndarray in [n_bands, psd_window//2+1]
        Power spectra values
    * freqs : ndarray in [n_bands, psd_window//2+1]
        Frequencies for FFT points
    * norm_vec : ndarray in [n_bands, 1]
        Normalization factor of each octave band

    """
    spectra = np.zeros((octave_band_env.shape[0], psd_window//2+1))
    freqs = np.zeros_like(spectra)

    for b in range(octave_band_env.shape[0]):
        # psd default: hanning window with no overlap
        if window_type == 'none':
            spectra[b, :], freqs[b, :] = psd(octave_band_env[b, :], NFFT=psd_window, Fs=hz, window=window_none)
        elif window_type == 'hanning':
            spectra[b, :], freqs[b, :] = psd(octave_band_env[b, :], NFFT=psd_window, Fs=hz, window=window_hanning)
        else:
            raise ValueError

    if norm == 'spec_max':
        # this method doesn't work well when the signal is short
        norm_vec = np.max(spectra, axis=-1, keepdims=True)  # scale to [0,1]
    elif norm == 'env_amp_square':
        norm_vec = (np.mean(octave_band_env, axis=-1, keepdims=True) ** 2)*(psd_window/hz)  # scale based on mean amplitude of the envelope
    elif norm == 'none':
        norm_vec = np.ones((spectra.shape[0], 1))
    else:
        raise ValueError

    spectra = spectra / norm_vec

    return spectra, freqs, norm_vec

def downsampleBands(octave_band_audio, hz, downsample_factor):
    """
    Downsample audio by integer factor
    Input
    -----
    * octave_band_audio : ndarray in [n_bands, N]
        Array of octave band filtered audio samples
    * hz : float or int
        Original audio sample rate in hertz
    * downsample_factor : int
        Factor to downsample audio by
    Output
    ------
    * ds_audio : ndarray in [n_bands, N]
        Downsampled audio array
    * hz_down : int
        Downsampled audio sample rate in hertz
    """

    # calculate downsampled audio rate in hertz
    downsample_factor = int(downsample_factor)  # factor must be integer
    hz_down = int(hz / downsample_factor)
    dsAudio = sig.decimate(octave_band_audio, downsample_factor, ftype='fir')

    return dsAudio, hz_down


def octaveBandFilter(audio, hz,
                     octave_bands=[125, 250, 500, 1000, 2000, 4000, 6000],
                     butter_ord=6):
    """
    Octave band filter raw audio. The audio is filtered through butterworth
    filters of order 6 (by default).
    Input
    -----
    * audio : ndarray
        Array of the raw audio
    * hz : float or int
        Audio sample rate in hertz
    * octave_bands : array-like
        list or array of octave band center frequencies
    * butter_ord: int
        butterworth filter order

    Output
    ------
    * octave_band_audio : ndarray
        Octave band filtered audio
    """
    octave_bands = sorted(octave_bands)
    octave_band_audio = np.zeros((len(octave_bands), audio.shape[-1]))
    # process each octave band
    for n, f in enumerate(octave_bands):

        # the band-pass butterworth doesn't work right when the filter order is high (above 3).
        if f < max(octave_bands):
            # filter the output at the octave band f
            f1 = f / sqrt(2)
            f2 = f * sqrt(2)
            b1, a1 = sig.butter(butter_ord, f1, btype='high', fs=hz)
            b2, a2 = sig.butter(butter_ord, f2, btype='low', fs=hz)

            out = sig.lfilter(b1, a1, audio)  # high-pass raw audio at f1
            out = sig.lfilter(b2, a2, out)  # low-pass after high-pass at f1

        else:
            b1, a1 = sig.butter(butter_ord, f, btype='high', fs=hz)
            out = sig.lfilter(b1, a1, audio)

        # stack-up octave band filtered audio
        octave_band_audio[n, :] = out

    return octave_band_audio

def hilbert_env(octave_band_audio):
    """
    Hilbert Envelope Extraction
    """
    analytic_signal = sig.hilbert(octave_band_audio, axis=-1)
    return np.abs(analytic_signal)

def syllable_rate_estimation(reference, hz):
    """
    Estimate speech syllable rate from the reference speech
    Input
    -----
    * reference : ndarray in [N] or [1, N]
        Array of the reference audio
    * hz : float or int
        Audio sample rate in hertz

    Output
    ------
    * sr_est : float
        The estimated syllable rate
    """

    bands = [125, 250, 500, 1000, 2000, 4000, 6000]
    ref_octave_bands = octaveBandFilter(reference, hz, octave_bands=bands)
    ref_octave_env = hilbert_env(ref_octave_bands)

    # low_pass 25 herz, then down sample to 100 herz
    target_sr = 100
    lp_b, lp_a = sig.butter(2, 25, btype='low', fs=hz)
    octaveEnv_lp = sig.lfilter(lp_b, lp_a, ref_octave_env)  # low-pass
    octaveEnv_lp, hz_down = downsampleBands(octaveEnv_lp, hz, hz / target_sr)
    assert (target_sr == hz_down)

    # band selection
    topM = 4
    octaveEnergy = np.sum(octaveEnv_lp ** 2, axis=-1)
    selected_band_idx = np.argpartition(octaveEnergy, -topM)[-topM:]
    selectedOctaveEnv = octaveEnv_lp[selected_band_idx, :]
    temporal_correlation = np.zeros_like(selectedOctaveEnv)

    # temporal correlation
    K = 11  # 110 ms
    half_frame = K // 2
    zero_pad = np.zeros((selectedOctaveEnv.shape[0], half_frame))
    selectedOctaveEnv = np.concatenate((zero_pad, selectedOctaveEnv, zero_pad), axis=-1)
    w = np.expand_dims(sig.windows.hann(K), axis=0)

    for t in range(temporal_correlation.shape[1]):
        temporal_correlation[:, t] = np.sqrt(
            np.abs(selectedOctaveEnv[:, t + half_frame] * np.sum(w * selectedOctaveEnv[:, t:t + K], axis=-1) / K))


    # spectral correlation
    crossband_correlation = np.zeros_like(temporal_correlation[0, :])
    all_iter = itertools.combinations(range(temporal_correlation.shape[0]), 2)
    all_iter = list(all_iter)
    for _, (b_idx1, b_idx2) in enumerate(all_iter):
        crossband_correlation = crossband_correlation + np.sqrt(
            temporal_correlation[b_idx1, :] * temporal_correlation[b_idx2, :])

    n_combination = len(all_iter)
    crossband_correlation = crossband_correlation / n_combination

    # normalize to [0, 1]
    crossband_correlation = crossband_correlation / np.max(crossband_correlation)
    # peak finding
    peaks, properties = sig.find_peaks(crossband_correlation, prominence=0.07)
    # calculate syllable rate
    sr_est = len(peaks) / (reference.shape[-1] / hz)

    return sr_est


def sram(degraded, hz, reference=None, sr=None, use_sr_estimation=False,
         window_duration_s: float = 1, downsample: int = None):
    """
    Calculate the Syllable-Rate-Adjusted-Modulation (SRAM) Index
    from the degraded audio
    Input
    -----
    * degraded : array-like
        Degraded audio sample
    * hz : int
        Audio sample rate in hertz
    * reference : array-like
        Clean reference audio sample, only required if use_sr_estimation is True
    * sr : float
        The syllable rate of the degraded audio only required if use_sr_estimation is False
    * use_sr_estimation : array-like
        If to use the build in sr estimation, when this is True, reference and sr input will be ignored
    * window_duration_s : float
        The window duration in s to calcuate the modulation power spectrum
    * downsample : int or None
        Downsampling integer factor

    Output
    ------
    * index : float
        The calculated index
    """
    bands = [125, 250, 500, 1000, 2000, 4000, 6000]
    # octave band filtering
    degr_octave_bands = octaveBandFilter(degraded, hz, octave_bands=bands)
    # envelope extraction
    degr_octave_env = hilbert_env(degr_octave_bands)

    # downsampling, if desired
    if downsample is not None:
        degr_octave_env, hz_down = downsampleBands(degr_octave_env, hz, downsample)
    else:
        hz_down = hz

    # obtain syllable rate value
    if use_sr_estimation:
        if reference is None:
            raise ValueError
        else:
            sr_est = syllable_rate_estimation(reference, hz)
        min_mf = floor(sr_est)
    else:
        sr_est = None
        if sr is None or sr <= 1:
            # fall back value if no sr value is given or sr is less than 1
            min_mf = 1
        else:
            min_mf = floor(sr)
    max_mf = 25.0

    # smse calc procedure
    len_env_window = round(window_duration_s * hz_down)
    spectras, sfreqs, norm_vec = octaveBandSpectra(degr_octave_env, hz_down, psd_window=len_env_window,
                                                   window_type='none', norm='env_amp_square')

    idx_to_include = np.bitwise_and(sfreqs[0, :] >= min_mf, sfreqs[0, :] <= max_mf)
    mod_sum_energy = np.sum(spectras[:, idx_to_include], axis=-1)
    index = np.mean(mod_sum_energy, axis=0)

    return index, sr_est