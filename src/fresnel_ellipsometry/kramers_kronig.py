"""Kramers--Kronig transforms used by the dielectric models.

The implementation below evaluates the differential Kramers--Kronig relation
with the Maclaurin even/odd summation scheme described by Ohta and Ishida,
Applied Spectroscopy 42, 952 (1988), doi:10.1366/0003702884430380.

The numerical convention is intentionally compatible with ``pyElli``'s
``elli.kkr.kkr.im2re`` routine, which was used during development of the
original research code.  This implementation is written locally so that the
package does not require pyElli at runtime.

Reference implementation consulted during validation:
https://github.com/PyEllips/pyElli

Notes
-----
This is the *differential* form: the returned real part approaches zero at
infinite frequency, so an independent high-frequency/background offset must
be supplied by the optical model.  The transform is sensitive to truncation
and spectral sampling; the fitting code therefore evaluates it on a wider,
uniform energy grid than the measured interval.
"""

from __future__ import annotations

import numpy as np


def im2re(imag: np.ndarray, energy: np.ndarray) -> np.ndarray:
    r"""Transform an imaginary response into its differential real response.

    The continuous relation is

    .. math::

        \Delta\epsilon_1(E_i)
        = \frac{2}{\pi}\,\mathcal P\int_0^\infty
          \frac{E\,\epsilon_2(E)}{E^2-E_i^2}\,dE.

    A Maclaurin even/odd discretization avoids evaluating the singular term at
    ``E == E_i`` by summing odd grid points for even output indices and vice
    versa.  For the uniform energy grids used by this project, this reproduces
    the numerical convention used by pyElli's ``im2re`` implementation.

    Parameters
    ----------
    imag:
        Imaginary part sampled on ``energy``.
    energy:
        Monotonically increasing energy axis.  At least two points are
        required.  The method is intended for an approximately uniform grid.

    Returns
    -------
    numpy.ndarray
        Differential real response on the same grid.
    """
    imag = np.asarray(imag, dtype=float)
    energy = np.asarray(energy, dtype=float)

    if imag.ndim != 1 or energy.ndim != 1:
        raise ValueError("imag and energy must be one-dimensional arrays.")
    if imag.size != energy.size:
        raise ValueError(
            "imag and energy must have the same length, "
            f"but have lengths {imag.size} and {energy.size}."
        )
    if energy.size < 2:
        raise ValueError("At least two energy points are required for the KK transform.")
    if not np.all(np.isfinite(imag)) or not np.all(np.isfinite(energy)):
        raise ValueError("imag and energy must contain only finite values.")
    if np.any(np.diff(energy) <= 0):
        raise ValueError("energy must be strictly increasing.")

    # Match the Maclaurin quadrature convention used during development,
    # including pyElli's endpoint interval construction. The fitting code
    # supplies a uniformly sampled energy grid.
    interval = np.diff(energy, prepend=energy[1] - energy[0])
    transformed = np.empty_like(imag, dtype=float)

    even = np.arange(0, energy.size, 2)
    odd = np.arange(1, energy.size, 2)

    if odd.size:
        x = energy[odd][None, :]
        y = imag[odd][None, :]
        xi = energy[even][:, None]
        transformed[even] = np.sum(x * y / (x * x - xi * xi), axis=1)
    else:
        transformed[even] = 0.0

    if even.size:
        x = energy[even][None, :]
        y = imag[even][None, :]
        xi = energy[odd][:, None]
        transformed[odd] = np.sum(x * y / (x * x - xi * xi), axis=1)

    return (4.0 / np.pi) * interval * transformed
