import numpy as np
import obspy
from obspy.clients.fdsn import Client


def get_labeled_window(x, batch_size):
    return x[:, :-batch_size], x[:, -batch_size:]


def make_windows(x, window_size, batch_size):
    window_step = np.expand_dims(np.arange(window_size + batch_size), axis=0)
    window_indexes = (
        window_step
        + np.expand_dims(np.arange(len(x) - (window_size + batch_size - 1)), axis=0).T
    )

    windowed_array = x[window_indexes]
    windows, labels = get_labeled_window(windowed_array, batch_size)

    return windows, labels


def convert_to_original(windowed_array):
    original = windowed_array[:, 0]
    original = np.append(original, windowed_array[-1, 1:])
    return original


def get_stream(start_time, duration, channel="HHE", station="GAZ", network="KOERI"):
    client = Client(network)
    t = obspy.UTCDateTime(start_time)
    st = client.get_waveforms("*", station, "*", channel, t, t + duration)

    return st
