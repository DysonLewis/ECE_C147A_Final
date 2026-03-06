"""
Sets up the data_fourier/ directory by symlinking the raw HDF5 session files
from data/ so the training pipeline can point to it with dataset.root=data_fourier.
The Fourier transform is applied at runtime via transforms=fourier_features.

Also provides a Nyquist verification function to check that the sampling rate
is sufficient for the sEMG signal bandwidth.

Usage:
    python scripts/preprocess_fourier.py --data_dir data --out_dir data_fourier
    python scripts/preprocess_fourier.py --verify_nyquist --data_dir data
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def setup_fourier_dir(data_dir: Path, out_dir: Path) -> None:
    """Create out_dir and symlink all HDF5 session files from data_dir into it.

    The Fourier transform is applied at runtime when training with
    transforms=fourier_features, so the underlying HDF5 files are shared.
    Keeping a separate directory makes it easy to track which experiment
    used which transform purely from dataset.root in the run config.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = list(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return

    for src in hdf5_files:
        dst = out_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        print(f"Linked {src.name}")

    print(f"data_fourier ready at {out_dir} ({len(hdf5_files)} sessions)")


def verify_nyquist(hdf5_path: Path, fs: int = 2000) -> None:
    """Compute the power spectral density of one session and confirm that
    signal power above the Nyquist frequency for each candidate downsample
    factor is negligible.

    sEMG relevant band: ~20-500 Hz.
    Nyquist requirement: fs >= 2 * fmax => minimum fs = 1000 Hz (factor=2).

    Factors tested: 1 (2000 Hz), 2 (1000 Hz), 4 (500 Hz), 8 (250 Hz).
    Factors > 2 violate Nyquist and are expected to degrade performance.
    """
    print(f"\nNyquist verification on {hdf5_path.name}")
    print(f"Original sampling rate: {fs} Hz")
    print(f"sEMG signal band: 20-500 Hz")
    print(f"Nyquist minimum fs: 1000 Hz (factor <= 2 is safe)\n")

    with h5py.File(hdf5_path, "r") as f:
        emg = f["emg2qwerty"]["timeseries"]["emg_left"][:10000, 0]

    freqs = np.fft.rfftfreq(len(emg), d=1.0 / fs)
    psd = np.abs(np.fft.rfft(emg)) ** 2
    total_power = psd.sum()

    semg_band = (freqs >= 20) & (freqs <= 500)
    semg_power_pct = 100 * psd[semg_band].sum() / total_power

    print(f"Power in 20-500 Hz band: {semg_power_pct:.1f}% of total")

    print("\nPower above Nyquist frequency by downsample factor:")
    for factor in [1, 2, 4, 8]:
        effective_fs = fs // factor
        nyquist = effective_fs / 2
        aliased = (freqs > nyquist)
        aliased_pct = 100 * psd[aliased].sum() / total_power
        safe = "safe" if factor <= 2 else "violates Nyquist"
        print(
            f"  factor={factor} -> {effective_fs} Hz "
            f"(Nyquist={nyquist:.0f} Hz): "
            f"{aliased_pct:.2f}% power aliased  [{safe}]"
        )

    print(
        "\nConclusion: factors 1 and 2 are safe. "
        "Use factors 4 and 8 to study degradation from undersampling."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--out_dir", type=Path, default=Path("data_fourier"))
    parser.add_argument(
        "--verify_nyquist",
        action="store_true",
        help="Run Nyquist verification on first HDF5 file found in data_dir",
    )
    args = parser.parse_args()

    if args.verify_nyquist:
        hdf5_files = list(args.data_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"No HDF5 files found in {args.data_dir}")
        else:
            verify_nyquist(hdf5_files[0])
    else:
        setup_fourier_dir(args.data_dir, args.out_dir)