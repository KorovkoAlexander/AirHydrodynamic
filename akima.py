#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:03:12 2017

@author: alex
"""

import numpy

def interpolate(x, y, x_new, axis=-1, out=None):
    x = numpy.array(x, dtype=numpy.float64, copy=True)
    y = numpy.array(y, dtype=numpy.float64, copy=True)
    xi = numpy.array(x_new, dtype=numpy.float64, copy=True)

    if axis != -1 or out is not None or y.ndim != 1:
        raise NotImplementedError("implemented in C extension module")

    if x.ndim != 1 or xi.ndim != 1:
        raise ValueError("x-arrays must be one dimensional")

    n = len(x)
    if n < 3:
        raise ValueError("array too small")
    if n != y.shape[axis]:
        raise ValueError("size of x-array must match data shape")

    dx = numpy.diff(x)
    if any(dx <= 0.0):
        raise ValueError("x-axis not valid")

    if any(xi < x[0]) or any(xi > x[-1]):
        raise ValueError("interpolation x-axis out of bounds")

    m = numpy.diff(y) / dx
    mm = 2.0 * m[0] - m[1]
    mmm = 2.0 * mm - m[0]
    mp = 2.0 * m[n - 2] - m[n - 3]
    mpp = 2.0 * mp - m[n - 2]

    m1 = numpy.concatenate(([mmm], [mm], m, [mp], [mpp]))

    dm = numpy.abs(numpy.diff(m1))
    f1 = dm[2:n + 2]
    f2 = dm[0:n]
    f12 = f1 + f2

    ids = numpy.nonzero(f12 > 1e-9 * numpy.max(f12))[0]
    b = m1[1:n + 1]

    b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
    c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
    d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2

    bins = numpy.digitize(xi, x)
    bins = numpy.minimum(bins, n - 1) - 1
    bb = bins[0:len(xi)]
    wj = xi - x[bb]

    return ((wj * d[bb] + c[bb]) * wj + b[bb]) * wj + y[bb]