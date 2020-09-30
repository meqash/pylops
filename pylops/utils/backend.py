import numpy as np
from scipy.signal import convolve, fftconvolve, oaconvolve
from pylops.utils import deps

if deps.cupy_enabled:
    import cupy as cp

if deps.cusignal_enabled:
    import cusignal

cu_message = 'cupy package not installed. Use numpy arrays of ' \
             'install cupy.'

cusignal_message = 'cusignal package not installed. Use numpy arrays of' \
                   'install cusignal.'


def get_module(backend='numpy'):
    """Returns correct numerical module based on backend string

    Parameters
    ----------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if backend == 'numpy':
        ncp = np
    elif backend == 'cupy':
        ncp = cp
    else:
        raise ValueError('backend must be numpy or cupy')
    return ncp


def get_module_name(mod):
    """Returns name of numerical module based on input numerical module

    Parameters
    ----------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    Returns
    -------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    """
    if mod == np:
        backend = 'numpy'
    elif mod == cp:
        backend = 'cupy'
    else:
        raise ValueError('module must be numpy or cupy')
    return backend


def get_array_module(x):
    """Returns correct numerical module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if deps.cupy_enabled:
        return cp.get_array_module(x)
    else:
        return np


def get_convolve(x):
    """Returns correct fftconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return convolve

    if cp.get_array_module(x) == np:
        return convolve
    else:
        if deps.cusignal_enabled:
            return cusignal.convolution.convolve
        else:
            raise ModuleNotFoundError(cusignal_message)


def get_fftconvolve(x):
    """Returns correct fftconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return fftconvolve

    if cp.get_array_module(x) == np:
        return fftconvolve
    else:
        if deps.cusignal_enabled:
            return cusignal.convolution.fftconvolve
        else:
            raise ModuleNotFoundError(cusignal_message)


def get_oaconvolve(x):
    """Returns correct oaconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return oaconvolve

    if cp.get_array_module(x) == np:
        return oaconvolve
    else:
        raise NotImplementedError('oaconvolve not implemented in '
                                  'cupy/cusignal. Consider using a different'
                                  'option...')


def to_cupy_conditional(x, y):
    """Convert y to cupy array conditional to x being a cupy array

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array to evaluate
    y : :obj:`numpy.ndarray`
        Array to convert

    Returns
    -------
    y : :obj:`cupy.ndarray`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == cp and cp.get_array_module(y) == np:
            y = cp.asarray(y)
    return y
