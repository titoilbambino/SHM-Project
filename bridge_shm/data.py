"""Read synchronized bridge channels and construct recording-local endpoint windows."""

from dataclasses import dataclass

import h5py
import numpy as np
from torch.utils.data import ConcatDataset, Dataset


@dataclass
class Recording:
    acceleration: np.ndarray
    strain: np.ndarray


def highpass_strain(strain, sample_rate_hz, cutoff_hz, order):
    """Causally high-pass a single strain trace, starting at its initial level."""
    from scipy.signal import butter, sosfilt

    sections = butter(order, cutoff_hz, btype="highpass", fs=sample_rate_hz, output="sos")
    # Zero filter state on this centered trace assumes a constant pre-recording level.
    centered = strain.astype(np.float64) - float(strain[0])
    return sosfilt(sections, centered).astype(np.float32)


def load_recordings(path, names, *, target, sample_rate_hz, feature_count, target_unit):
    """Return float32 records in the sorted AL/AG channel order used by 001."""
    if len(set(names)) != len(names):
        raise ValueError("A recording appears more than once in the split")
    target_sensor, axis = target.rsplit("_", 1)
    records = {}
    channel_order = None
    with h5py.File(path, "r") as file:
        for name in names:
            group = file[name]
            gauge = group["strain"][target_sensor]
            if group.attrs["sample_rate"] != sample_rate_hz or gauge.attrs["unit"] != target_unit:
                raise ValueError(f"{name}: unexpected sample rate or target unit")
            channels = {
                f"{sensor}_{direction}": values
                for sensor, axes in group["acceleration"].items()
                for direction, values in axes.items()
                if sensor.startswith("AL") or (sensor.startswith("AG") and direction in ("y", "z"))
            }
            order = sorted(channels)
            if len(order) != feature_count or (channel_order is not None and order != channel_order):
                raise ValueError(f"{name}: acceleration channel layout differs")
            channel_order = order
            y = gauge[axis][:].astype(np.float32)
            x = np.empty((len(y), len(order)), dtype=np.float32)
            for col, key in enumerate(order):
                if channels[key].shape != y.shape:
                    raise ValueError(f"{name}/{key}: length differs from {target}")
                x[:, col] = channels[key][:]
            records[name] = Recording(x, y)
    return records


def fit_channel_mean(records, names):
    """Sample-weighted per-channel means of the specified fit recordings."""
    sums = [records[name].acceleration.sum(axis=0, dtype=np.float64) for name in names]
    count = sum(len(records[name].strain) for name in names)
    return (np.sum(sums, axis=0) / count).astype(np.float32)


def endpoint_indices(n_samples, window, stride):
    if window <= 0 or stride <= 0 or window > n_samples:
        raise ValueError(f"Invalid window {window} or stride {stride} for {n_samples} samples")
    return range(window - 1, n_samples, stride)


class RecordingWindows(Dataset):
    """Store one centered recording. Each item is a view ending at its target."""

    def __init__(self, record, mean, window, stride):
        self.ends = endpoint_indices(len(record.strain), window, stride)
        self.window = window
        self.x = record.acceleration - mean
        self.targets = record.strain[window - 1 :: stride]

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        stop = self.ends[index] + 1
        return self.x[stop - self.window : stop], self.targets[index]


def training_windows(records, names, mean, window, stride):
    return ConcatDataset([RecordingWindows(records[n], mean, window, stride) for n in names])
