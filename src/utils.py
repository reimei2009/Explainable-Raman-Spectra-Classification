"""
utils.py - Utility functions for the Raman Spectra Analysis project.

Provides helpers for logging, path resolution, directory management,
and dataset discovery.
"""

import logging
import os
import re
from pathlib import Path


def get_project_root() -> Path:
    """
    Return the absolute Path to the project root directory.

    The project root is defined as the parent directory of the ``src/``
    package (i.e., two levels up from this file).

    Returns
    -------
    Path
        Absolute path to the project root.

    Examples
    --------
    >>> root = get_project_root()
    >>> print(root)
    C:/Users/ADMIN/Desktop/Prj/KLTN
    """
    return Path(__file__).resolve().parent.parent


def ensure_dir(path) -> Path:
    """
    Create *path* (and all parents) if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The resolved path that was created (or already existed).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(log_file: str = "outputs/processing.log") -> logging.Logger:
    """
    Configure the root logger to write to both a file and the console.

    Parameters
    ----------
    log_file : str, optional
        Relative (to project root) or absolute path of the log file.
        Defaults to ``outputs/processing.log``.

    Returns
    -------
    logging.Logger
        Configured root logger.

    Notes
    -----
    The log format includes timestamp, module name, log level, and message.
    File handler uses DEBUG level; console handler uses INFO level.
    """
    project_root = get_project_root()

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = project_root / log_file

    ensure_dir(log_path.parent)

    log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(ch)

    logger.info("Logging initialised. Log file: %s", log_path)
    return logger


def list_samples(data_root) -> list:
    """
    Return a sorted list of sample folder names found directly under *data_root*.

    Only directories whose names match the expected pattern
    ``[FfMm]<digits><suffix>`` (e.g. ``M78S``, ``f41p``) are included.

    Parameters
    ----------
    data_root : str or Path
        Root directory that contains one sub-folder per sample.

    Returns
    -------
    list of str
        Sorted sample folder names.

    Examples
    --------
    >>> samples = list_samples(r"Data/different potential")
    >>> print(samples[:3])
    ['F35S', 'F42S', 'F45S']
    """
    data_root = Path(data_root)
    pattern = re.compile(r"^[FfMm]\d+[A-Za-z]+$")
    samples = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and pattern.match(d.name)
    )
    return samples


def get_potentials(sample_folder) -> list:
    """
    Return a sorted list of unique potential values (in mV) present in
    *sample_folder*, inferred from the ``.spc`` file names.

    File names are expected to follow the pattern ``<potential>_<replicate>.spc``
    where *potential* is an integer (possibly negative).

    Parameters
    ----------
    sample_folder : str or Path
        Path to a single sample directory.

    Returns
    -------
    list of int
        Sorted unique potential values in mV.

    Examples
    --------
    >>> pots = get_potentials(r"Data/different potential/M78S")
    >>> print(pots)
    [-400, -300, -200, -100, 0, 100, 200, 300, 400]
    """
    folder = Path(sample_folder)
    pattern = re.compile(r"^(-?\d+)_\d+\.spc$", re.IGNORECASE)
    potentials = set()
    for f in folder.iterdir():
        m = pattern.match(f.name)
        if m:
            potentials.add(int(m.group(1)))
    return sorted(potentials)
