import numpy as np


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def compute_isi_histograms_from_spiketrain(spike_train: np.ndarray, max_time: int,
                                bin_size: int, sampling_f: float):
    """
    Computes the Inter-Spike Intervals histogram from a given spike train.

    This implementation only works if you have numba installed, to accelerate the
    computation time.

    Parameters
    ----------
    spike_train: np.ndarray
        The ordered spike train to compute the ISI.
    max_time: int
        Compute the ISI from 0 to +max_time (in sampling time).
    bin_size: int
        Size of a bin (in sampling time).
    sampling_f: float
        Sampling rate/frequency (in Hz).

    Returns
    -------
    tuple (ISI, bins)
    ISI: np.ndarray[int64]
        The computed ISI histogram.
    bins: np.ndarray[float64]
        The bins for the ISI histogram.
    """
    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_ISI_from_spiketrain cannot run without numba.")
        return 0

    return _compute_isi_histograms_from_spiketrain(spike_train.astype(np.int64), max_time, bin_size, sampling_f)

if HAVE_NUMBA:
    @numba.jit((numba.int64[::1], numba.int32, numba.int32, numba.float32),
               nopython=True, nogil=True, cache=True)
    def _compute_isi_histograms_from_spiketrain(spike_train, max_time, bin_size, sampling_f):
        n_bins = int(max_time / bin_size)

        bins = np.arange(0, max_time+bin_size, bin_size) * 1e3 / sampling_f
        ISI = np.zeros(n_bins, dtype=np.int64)

        for i in range(1, len(spike_train)):
            diff = spike_train[i] - spike_train[i-1]

            if diff >= max_time:
                continue

            bin = int(diff / bin_size)
            ISI[bin] += 1

        return ISI, bins



def compute_isi_histograms(sorting, window_ms: float = 50.0, bin_ms: float = 1.0,
                method: str = "auto"):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.
    """

    assert method in ("auto", "numba", "numpy")

    if method == "auto":
        method = "numba" if HAVE_NUMBA else "numpy"

    if method == "numpy":
        return compute_isi_histograms_numpy(sorting, window_ms, bin_ms)
    if method == "numba":
        return compute_isi_histograms_numba(sorting, window_ms, bin_ms)

def compute_isi_histograms_numpy(sorting, window_ms: float = 50.0, bin_ms: float = 1.0):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.

    This is a very standard numpy implementation, nothing fancy.

    Implementation: Aurélien Wyngaard
    """
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)

    window_size = int(round(fs * window_ms * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = int(window_size / bin_size)
    assert num_bins >= 1

    ISIs = np.zeros((num_units, num_bins), dtype=np.int64)
    bins = np.arange(0, window_size+bin_size, bin_size) * 1e3 / fs
    
    # TODO: There might be a better way than a double for loop?
    for i, unit_id in enumerate(sorting.unit_ids):
        for seg_index in range(sorting.get_num_segments()):
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=seg_index)
            ISI = np.histogram(np.diff(spike_train), bins=num_bins, range=(0, window_size-1))[0]
            ISIs[i] += ISI

    return ISIs, bins


def compute_isi_histograms_numba(sorting, window_ms: float = 50.0, bin_ms: float = 1.0):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.

    This is a "brute force" method using compiled code (numba)
    to accelerate the computation.

    Implementation: Aurélien Wyngaard
    """

    assert HAVE_NUMBA
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)

    window_size = int(round(fs * window_ms * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = int(window_size / bin_size)
    assert num_bins >= 1

    bins = np.arange(0, window_size+bin_size, bin_size) * 1e3 / fs
    spikes = sorting.get_all_spike_trains(outputs="unit_index")

    ISIs = np.zeros((num_units, num_bins), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        _compute_isi_histograms_numba(ISIs, spikes[seg_index][0].astype(np.int64),
                           spikes[seg_index][1].astype(np.int32),
                           window_size, bin_size, fs)

    return ISIs, bins


if HAVE_NUMBA:
    @numba.jit((numba.int64[:, ::1], numba.int64[::1], numba.int32[::1], numba.int32, numba.int32, numba.float32),
               nopython=True, nogil=True, cache=True, parallel=True)
    def _compute_isi_histograms_numba(ISIs, spike_trains, spike_clusters, max_time, bin_size, sampling_f):
        n_units = ISIs.shape[0]

        for i in numba.prange(n_units):
            spike_train = spike_trains[spike_clusters == i]

            ISIs[i] += _compute_isi_histograms_from_spiketrain(spike_train, max_time, bin_size, sampling_f)[0]