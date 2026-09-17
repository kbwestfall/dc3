"""
Bit masks, and arrays of them.

:class:`BitMask` handles the bit definitions and the operations on raw integer
values; :class:`BitMaskArray` pairs a mask array with its definition so that
flags can be read and set by name.

.. note::

    Adapted from ``pypeit/core/bitmask.py`` and
    ``pypeit/images/bitmaskarray.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause); see
    ``licenses/README.rst``.

    Two things differ.  **Bits are declared as a class-level mapping** of name to
    description, rather than passed to the constructor as parallel lists of keys
    and descriptions that must be kept in register -- the same change
    :class:`~dc3.par.parset.ParSet` makes for parameters.  And
    :class:`BitMaskArray` is a **plain array wrapper** rather than a
    ``DataContainer``, so it carries no file-format machinery; serializing it is
    the datamodel's job, not its own.

Declaring a mask
----------------

.. code-block:: python

    class SpectrumBitMask(BitMask):
        prefix = 'SPCBIT'
        bits = {
            'NODATA': 'Pixel has no data',
            'CR': 'Pixel contaminated by a cosmic ray',
        }

The bit *value* is the position in the mapping, so **order is the datamodel**:
inserting a bit anywhere but the end renumbers every bit after it, and silently
reinterprets every mask already written to a file.  To retire a bit, keep its
entry and set its description to None; it is then excluded from
:func:`BitMask.keys` but still occupies its position.

.. include:: ../include/links.rst
"""

import textwrap
from typing import ClassVar

from astropy.io import fits
import numpy as np

from ..pkg.exceptions import DC3BitMaskError, DC3CodingError


__all__ = ['BitMask', 'BitMaskArray']


class BitMask:
    """
    Base class for handling and manipulating bit masks.

    Subclasses declare their bits as a class-level mapping; see the module
    documentation.  Instances carry no state beyond the declaration, so the
    operations are all class methods and an instance is never required.

    Attributes
    ----------
    nbits : int
        Number of declared bits, including any retired ones.
    max_value : int
        The largest valid mask value given the number of bits.
    """

    prefix: ClassVar[str] = 'BIT'
    """Prefix used for the FITS header keywords recording the bit definitions."""

    bits: ClassVar[dict] = {}
    """
    Mapping of bit name to description, in bit order.

    The bit value is the position in this mapping.  A description of None marks
    a retired bit, which keeps its position but is otherwise ignored.
    """

    def __init__(self):
        if len(type(self).bits) > 64:
            raise DC3CodingError(
                f'{type(self).__name__} declares {len(type(self).bits)} bits; at most 64 are '
                'allowed, since the mask must fit in a 64-bit integer.'
            )
        self.nbits = len(type(self).bits)
        self.max_value = (1 << self.nbits) - 1

    @classmethod
    def bit(cls, flag):
        """
        Return the bit value of a named flag.

        Parameters
        ----------
        flag : str
            The bit name.

        Returns
        -------
        int
            The bit position.

        Raises
        ------
        DC3BitMaskError
            Raised if the flag is not declared, or has been retired.
        """
        if flag not in cls.bits:
            raise DC3BitMaskError(
                f'{flag} is not a bit of {cls.__name__}.  Valid bits are: {cls.keys()}.'
            )
        if cls.bits[flag] is None:
            raise DC3BitMaskError(
                f'{flag} is a retired bit of {cls.__name__} and cannot be used.  It is kept '
                'only so that the remaining bits do not change value.'
            )
        return list(cls.bits.keys()).index(flag)

    @classmethod
    def keys(cls):
        """
        Return the list of usable bit names, in bit order.

        Returns
        -------
        list
            Bit names, excluding any that have been retired.
        """
        return [key for key, descr in cls.bits.items() if descr is not None]

    @classmethod
    def _prep_flags(cls, flag):
        """
        Normalize a flag argument to a list of valid bit names.

        Parameters
        ----------
        flag : str, list, None
            One or more bit names.  If None, all usable bits are returned.

        Returns
        -------
        list
            The bit names.

        Raises
        ------
        DC3BitMaskError
            Raised if any name is not a usable bit.
        """
        if flag is None:
            return cls.keys()
        _flag = [flag] if isinstance(flag, str) else list(flag)
        unrecognized = [f for f in _flag if f not in cls.bits]
        if len(unrecognized) > 0:
            raise DC3BitMaskError(
                f'Unrecognized bits for {cls.__name__}: {unrecognized}.  '
                f'Valid bits are: {cls.keys()}.'
            )
        retired = [f for f in _flag if cls.bits[f] is None]
        if len(retired) > 0:
            raise DC3BitMaskError(f'Retired bits of {cls.__name__} cannot be used: {retired}.')
        return _flag

    @classmethod
    def minimum_dtype(cls, asuint=False):
        """
        Return the smallest integer type that can hold every declared bit.

        Parameters
        ----------
        asuint : bool, optional
            Return an unsigned type.  Signed types are returned by default.

        Returns
        -------
        type
            The :mod:`numpy` dtype.

        Notes
        -----
        A signed type of at least 16 bits is used even when 8 would do, because
        :mod:`astropy.io.fits` cannot write ``int8``.
        """
        nbits = len(cls.bits)
        if nbits < 8:
            return np.uint8 if asuint else np.int16
        if nbits < 16:
            return np.uint16 if asuint else np.int16
        if nbits < 32:
            return np.uint32 if asuint else np.int32
        return np.uint64 if asuint else np.int64

    @classmethod
    def flagged(cls, value, flag=None, exclude=None, and_not=None):
        """
        Determine whether any of the given bits is set.

        Parameters
        ----------
        value : int, :class:`numpy.ndarray`
            The mask value or values to test.
        flag : str, list, optional
            The bits to test.  If None, every usable bit is tested, i.e. the
            result reports whether the value is flagged at all.
        exclude : str, list, optional
            Bits to remove from those tested.  Useful for "anything but these".
        and_not : str, list, optional
            Bits that must *not* be set.  This supersedes ``flag``: a value with
            one of these bits set returns False even if it also has a bit named
            in ``flag``.

        Returns
        -------
        bool, :class:`numpy.ndarray`
            True where any of the selected bits is set.
        """
        _flag = cls._prep_flags(flag)
        if exclude is not None:
            _exclude = cls._prep_flags(exclude)
            _flag = [f for f in _flag if f not in _exclude]
        if len(_flag) == 0:
            return np.zeros_like(value, dtype=bool)

        out = (value & (1 << cls.bit(_flag[0]))) != 0
        for f in _flag[1:]:
            out = out | ((value & (1 << cls.bit(f))) != 0)

        if and_not is None:
            return out
        return out & np.logical_not(cls.flagged(value, flag=and_not))

    @classmethod
    def flagged_bits(cls, value):
        """
        Return the names of the bits set in a single mask value.

        Parameters
        ----------
        value : int
            A single mask value.

        Returns
        -------
        list
            The names of the bits that are set, in bit order.

        Raises
        ------
        DC3BitMaskError
            Raised if the value is not a single integer.
        """
        if not np.issubdtype(type(value), np.integer):
            raise DC3BitMaskError(f'Expected a single integer, got {type(value).__name__}.')
        if value <= 0:
            return []
        return [key for key in cls.keys() if (value & (1 << cls.bit(key))) != 0]

    @classmethod
    def turn_on(cls, value, flag):
        """
        Set one or more bits.

        Parameters
        ----------
        value : int, :class:`numpy.ndarray`
            The mask value or values to modify.
        flag : str, list
            The bits to set.

        Returns
        -------
        int, :class:`numpy.ndarray`
            The modified value, with the input dtype preserved.
        """
        out = value
        for f in cls._prep_flags(flag):
            out = out | (1 << cls.bit(f))
        return out.astype(value.dtype) if isinstance(value, np.ndarray) else out

    @classmethod
    def turn_off(cls, value, flag):
        """
        Clear one or more bits.

        Parameters
        ----------
        value : int, :class:`numpy.ndarray`
            The mask value or values to modify.
        flag : str, list
            The bits to clear.

        Returns
        -------
        int, :class:`numpy.ndarray`
            The modified value, with the input dtype preserved.
        """
        out = value
        for f in cls._prep_flags(flag):
            out = out & ~(1 << cls.bit(f))
        return out.astype(value.dtype) if isinstance(value, np.ndarray) else out

    @classmethod
    def toggle(cls, value, flag):
        """
        Invert one or more bits.

        Parameters
        ----------
        value : int, :class:`numpy.ndarray`
            The mask value or values to modify.
        flag : str, list
            The bits to toggle.

        Returns
        -------
        int, :class:`numpy.ndarray`
            The modified value, with the input dtype preserved.
        """
        out = value
        for f in cls._prep_flags(flag):
            out = out ^ (1 << cls.bit(f))
        return out.astype(value.dtype) if isinstance(value, np.ndarray) else out

    @classmethod
    def consolidate(cls, value, flag_set, consolidated_flag):
        """
        Set one bit wherever any of a set of bits is set.

        Parameters
        ----------
        value : :class:`numpy.ndarray`
            The mask values to modify.
        flag_set : str, list
            The bits to look for.
        consolidated_flag : str
            The bit to set where any of ``flag_set`` is found.

        Returns
        -------
        :class:`numpy.ndarray`
            The modified values.
        """
        indx = cls.flagged(value, flag=flag_set)
        value[indx] = cls.turn_on(value[indx], consolidated_flag)
        return value

    @classmethod
    def unpack(cls, value, flag=None):
        """
        Return one boolean array per bit.

        Parameters
        ----------
        value : :class:`numpy.ndarray`
            The mask values to unpack.
        flag : str, list, optional
            The bits to unpack.  If None, every usable bit is unpacked.

        Returns
        -------
        tuple
            One boolean array per requested bit, in the requested order.
        """
        return tuple(cls.flagged(value, flag=f) for f in cls._prep_flags(flag))

    @classmethod
    def info(cls):
        """Print the declared bits and their descriptions."""
        for key in cls.keys():
            print(f'         Bit: {key} = {cls.bit(key)}')
            print(textwrap.fill(f' Description: {cls.bits[key]}', 78))
            print(' ')

    @classmethod
    def to_dict(cls, prefix=None):
        """
        Return the bit definitions as a dictionary keyed for a FITS header.

        Parameters
        ----------
        prefix : str, optional
            Keyword prefix, overriding :attr:`prefix`.

        Returns
        -------
        dict
            Mapping of header keyword to bit name.
        """
        _prefix = cls.prefix if prefix is None else prefix
        ndig = len(str(max(len(cls.bits) - 1, 1)))
        return {f'{_prefix}{str(cls.bit(key)).zfill(ndig)}': key for key in cls.keys()}

    @classmethod
    def to_header(cls, hdr=None, prefix=None):
        """
        Record the bit definitions in a FITS header.

        Writing the definitions alongside the mask is what makes a stored mask
        interpretable later, and is what :func:`validate_header` checks against.

        Parameters
        ----------
        hdr : :class:`astropy.io.fits.Header`, optional
            Header to modify in place.  If None, a new header is created.
        prefix : str, optional
            Keyword prefix, overriding :attr:`prefix`.

        Returns
        -------
        :class:`astropy.io.fits.Header`
            The header including the bit definitions.
        """
        _hdr = fits.Header() if hdr is None else hdr
        for keyword, key in cls.to_dict(prefix=prefix).items():
            _hdr[keyword] = (key, cls.bits[key])
        return _hdr

    @classmethod
    def parse_header(cls, hdr, prefix=None):
        """
        Read bit definitions from a FITS header.

        Parameters
        ----------
        hdr : :class:`astropy.io.fits.Header`
            The header to read.
        prefix : str, optional
            Keyword prefix, overriding :attr:`prefix`.

        Returns
        -------
        dict
            Mapping of bit value to bit name, as recorded in the header.
        """
        _prefix = cls.prefix if prefix is None else prefix
        found = {}
        for keyword in hdr.keys():
            if not keyword.startswith(_prefix):
                continue
            suffix = keyword[len(_prefix):]
            if not suffix.isdigit():
                continue
            found[int(suffix)] = hdr[keyword]
        return found

    @classmethod
    def validate_header(cls, hdr, prefix=None):
        """
        Check that a header's bit definitions match this class.

        A mask value only means anything with respect to the definitions in
        force when it was written.  If a bit has been inserted, removed or
        renamed since, every stored value is silently reinterpreted, so the
        mismatch is reported rather than tolerated.

        Parameters
        ----------
        hdr : :class:`astropy.io.fits.Header`
            The header to check.
        prefix : str, optional
            Keyword prefix, overriding :attr:`prefix`.

        Raises
        ------
        DC3BitMaskError
            Raised if the header records no bits for this prefix, or records
            bits that disagree with the class declaration.
        """
        found = cls.parse_header(hdr, prefix=prefix)
        if len(found) == 0:
            raise DC3BitMaskError(
                'Header records no bit definitions with the prefix '
                f'{cls.prefix if prefix is None else prefix}, so its mask values cannot be '
                'interpreted.'
            )
        expected = {cls.bit(key): key for key in cls.keys()}
        if found != expected:
            differences = [
                f'bit {value}: file has {found.get(value, "nothing")!r}, '
                f'{cls.__name__} has {expected.get(value, "nothing")!r}'
                for value in sorted(set(found) | set(expected))
                if found.get(value) != expected.get(value)
            ]
            raise DC3BitMaskError(
                f'Bit definitions in the file do not match {cls.__name__}: '
                + '; '.join(differences)
                + '.  Mask values written under different definitions mean different things.'
            )


class BitMaskArray:
    """
    An integer mask array paired with the :class:`BitMask` that defines it.

    Flags are read and set by name, which reads far better than repeated calls
    passing the flag as a string:

    .. code-block:: python

        mask.CR                      # boolean array of cosmic-ray pixels
        mask.turn_on('CR', gpm)      # set the bit where gpm is True

    Subclasses declare the bit definition they use:

    .. code-block:: python

        class SpectrumMask(BitMaskArray):
            bitmask = SpectrumBitMask

    This is deliberately a plain wrapper: it holds an array and knows what its
    bits mean, and nothing else.  Reading and writing it belongs to the
    datamodel.

    Parameters
    ----------
    shape : tuple, :class:`numpy.ndarray`
        The shape of a new, zeroed mask, or an existing integer array to adopt.
    asuint : bool, optional
        Use an unsigned integer type for a newly created array.

    Attributes
    ----------
    mask : :class:`numpy.ndarray`
        The raw integer mask values.
    """

    bitmask: ClassVar[type] = None
    """The :class:`BitMask` subclass defining the bits of this array."""

    def __init__(self, shape, asuint=False):
        if type(self).bitmask is None:
            raise DC3CodingError(
                f'{type(self).__name__} does not declare a bitmask, so its values have no '
                'meaning.  Set the bitmask class attribute to a BitMask subclass.'
            )
        if isinstance(shape, np.ndarray):
            if not np.issubdtype(shape.dtype, np.integer):
                raise DC3BitMaskError(
                    f'A mask array must have an integer type, not {shape.dtype}.'
                )
            self.mask = shape
        else:
            self.mask = np.zeros(shape, dtype=type(self).bitmask.minimum_dtype(asuint=asuint))

    def __getattr__(self, name):
        """
        Return the boolean array for a bit accessed by name.

        Only called when normal attribute lookup fails, so it never shadows a
        real attribute or method.
        """
        bitmask = type(self).bitmask
        if bitmask is not None and name in bitmask.bits:
            return bitmask.flagged(self.mask, flag=name)
        raise AttributeError(f'{type(self).__name__} has no attribute {name!r}.')

    def __getitem__(self, item):
        """Index the underlying mask array."""
        return self.mask[item]

    def __setitem__(self, item, value):
        """Assign into the underlying mask array."""
        self.mask[item] = value

    @property
    def shape(self):
        """The shape of the mask array."""
        return self.mask.shape

    @property
    def dtype(self):
        """The dtype of the mask array."""
        return self.mask.dtype

    def flagged(self, flag=None, exclude=None, and_not=None):
        """
        Determine where any of the given bits is set.

        Parameters
        ----------
        flag : str, list, optional
            The bits to test.  If None, every usable bit is tested.
        exclude : str, list, optional
            Bits to remove from those tested.
        and_not : str, list, optional
            Bits that must not be set.

        Returns
        -------
        :class:`numpy.ndarray`
            True where any of the selected bits is set.
        """
        return type(self).bitmask.flagged(self.mask, flag=flag, exclude=exclude, and_not=and_not)

    def turn_on(self, flag, select=None):
        """
        Set one or more bits, optionally only where selected.

        Parameters
        ----------
        flag : str, list
            The bits to set.
        select : :class:`numpy.ndarray`, optional
            Boolean array or index selecting where to set them.  If None, the
            bits are set everywhere.
        """
        # NOTE: None cannot be passed through as the index: numpy reads it as
        # np.newaxis and would add an axis rather than select everything.
        _select = Ellipsis if select is None else select
        self.mask[_select] = type(self).bitmask.turn_on(self.mask[_select], flag)

    def turn_off(self, flag, select=None):
        """
        Clear one or more bits, optionally only where selected.

        Parameters
        ----------
        flag : str, list
            The bits to clear.
        select : :class:`numpy.ndarray`, optional
            Boolean array or index selecting where to clear them.  If None, the
            bits are cleared everywhere.
        """
        # NOTE: see turn_on for why None is converted rather than passed through.
        _select = Ellipsis if select is None else select
        self.mask[_select] = type(self).bitmask.turn_off(self.mask[_select], flag)

    def flagged_bits(self, index):
        """
        Return the names of the bits set at one element.

        Parameters
        ----------
        index : int, tuple
            Index of the element to report.

        Returns
        -------
        list
            The names of the bits that are set there.
        """
        return type(self).bitmask.flagged_bits(self.mask[index])

    def to_header(self, hdr=None, prefix=None):
        """
        Record the bit definitions in a FITS header.

        Parameters
        ----------
        hdr : :class:`astropy.io.fits.Header`, optional
            Header to modify in place.  If None, a new header is created.
        prefix : str, optional
            Keyword prefix, overriding the bitmask's own.

        Returns
        -------
        :class:`astropy.io.fits.Header`
            The header including the bit definitions.
        """
        return type(self).bitmask.to_header(hdr=hdr, prefix=prefix)
