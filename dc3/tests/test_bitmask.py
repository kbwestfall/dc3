"""
Tests for :class:`~dc3.core.bitmask.BitMask` and
:class:`~dc3.core.bitmask.BitMaskArray`.
"""

from astropy.io import fits
import numpy as np
import pytest

from dc3.core.bitmask import RETIRED_BIT_PREFIX, BitMask, BitMaskArray
from dc3.pkg.exceptions import DC3BitMaskError, DC3CodingError


class ExampleBitMask(BitMask):
    """A mask with a retired bit, to exercise the value-stability rule."""

    prefix = 'EXBIT'
    bits = {
        'NODATA': 'Pixel has no data',
        'CR': 'Pixel contaminated by a cosmic ray',
        'OLDFLAG': f'{RETIRED_BIT_PREFIX} Pixel was rejected by a since-removed test',
        'LOWSNR': 'Pixel has low signal-to-noise',
    }


class ExampleMaskArray(BitMaskArray):
    """A mask array using the example definition."""

    bitmask = ExampleBitMask


# ----------------------------------------------------------------------
# Declaration
# ----------------------------------------------------------------------
def test_bit_values_follow_declaration_order():
    """The bit value is the position in the declaration."""
    assert ExampleBitMask.bit('NODATA') == 0, 'First declared bit is not bit 0'
    assert ExampleBitMask.bit('CR') == 1, 'Second declared bit is not bit 1'
    assert ExampleBitMask.bit('LOWSNR') == 3, \
        'A retired bit did not hold its position; the bits after it shifted'


def test_retired_bits_are_excluded_but_hold_their_place():
    """
    A retired bit keeps its value so that stored masks stay interpretable.

    Removing it outright would renumber every later bit and silently
    reinterpret every mask already written to a file.
    """
    assert ExampleBitMask.keys() == ['NODATA', 'CR', 'LOWSNR'], \
        'Retired bit was not excluded from the usable keys'
    assert ExampleBitMask.all_keys() == ['NODATA', 'CR', 'OLDFLAG', 'LOWSNR'], \
        'Retired bit was dropped from the full layout rather than kept in place'
    assert ExampleBitMask().nbits == 4, \
        'nbits should count the retired bit, since it still occupies a position'
    assert ExampleBitMask.is_retired('OLDFLAG'), 'Retired bit was not recognized as retired'
    assert not ExampleBitMask.is_retired('CR'), 'A live bit was reported as retired'
    with pytest.raises(DC3BitMaskError, match='Retired'):
        ExampleBitMask.flagged(np.zeros(3, dtype=int), flag='OLDFLAG')


def test_retired_bit_keeps_its_description():
    """
    Retiring a bit must not discard what it meant.

    A mask written before the bit was retired still has it set, so reading that
    file requires knowing what the bit recorded.
    """
    descr = ExampleBitMask.bits['OLDFLAG']
    assert descr.startswith(RETIRED_BIT_PREFIX), 'Retirement is not marked in the description'
    assert 'since-removed test' in descr, \
        'The original meaning of the bit was lost when it was retired'


def test_bit_reports_the_position_of_a_retired_bit():
    """
    The position of a retired bit is still a fact about the layout.

    Whether a bit may be *used* is enforced where it is used, not here, so that
    the header machinery can record the full layout.
    """
    assert ExampleBitMask.bit('OLDFLAG') == 2, \
        'A retired bit did not report its position, which the header record needs'


def test_unknown_bit_is_rejected():
    """An undeclared bit name is an error, not a silently ignored no-op."""
    with pytest.raises(DC3BitMaskError, match='not a bit of'):
        ExampleBitMask.bit('NO_SUCH_BIT')
    with pytest.raises(DC3BitMaskError, match='Unrecognized'):
        ExampleBitMask.flagged(np.zeros(3, dtype=int), flag='NO_SUCH_BIT')


def test_too_many_bits_is_a_coding_error():
    """More than 64 bits cannot fit in the mask value."""
    class TooMany(BitMask):
        bits = {f'BIT{i}': f'Bit {i}' for i in range(65)}

    with pytest.raises(DC3CodingError, match='at most 64'):
        TooMany()


def test_minimum_dtype():
    """The smallest usable integer type is chosen, avoiding int8."""
    assert ExampleBitMask.minimum_dtype() == np.int16, \
        'A four-bit mask should use int16, since astropy cannot write int8'
    assert ExampleBitMask.minimum_dtype(asuint=True) == np.uint8, \
        'An unsigned four-bit mask should use uint8'


# ----------------------------------------------------------------------
# Operations on raw values
# ----------------------------------------------------------------------
def test_turn_on_off_and_toggle():
    """Bits can be set, cleared and inverted, preserving the dtype."""
    value = np.zeros(4, dtype=np.int16)
    value = ExampleBitMask.turn_on(value, 'CR')
    assert np.all(value == 2), 'Setting bit 1 did not give a value of 2'
    assert value.dtype == np.int16, 'Setting a bit changed the array dtype'

    value = ExampleBitMask.turn_on(value, ['NODATA', 'LOWSNR'])
    assert np.all(value == 1 + 2 + 8), 'Setting several bits at once gave the wrong value'

    value = ExampleBitMask.turn_off(value, 'CR')
    assert np.all(value == 1 + 8), 'Clearing a bit gave the wrong value'

    value = ExampleBitMask.toggle(value, 'CR')
    assert np.all(value == 1 + 2 + 8), 'Toggling an unset bit did not set it'


def test_flagged_variants():
    """flagged supports any-of, all-but, and and-not selections."""
    value = np.array([0, 1, 2, 3], dtype=np.int16)   # none, NODATA, CR, both
    assert np.array_equal(
        ExampleBitMask.flagged(value), [False, True, True, True]
    ), 'Flagging with no bits named did not report "flagged at all"'
    assert np.array_equal(
        ExampleBitMask.flagged(value, flag='CR'), [False, False, True, True]
    ), 'Flagging a single bit gave the wrong result'
    assert np.array_equal(
        ExampleBitMask.flagged(value, exclude='NODATA'), [False, False, True, True]
    ), 'Excluding a bit did not remove it from those tested'
    assert np.array_equal(
        ExampleBitMask.flagged(value, and_not='CR'), [False, True, False, False]
    ), 'and_not did not suppress values carrying the excluded bit'


def test_flagged_bits_and_unpack():
    """The set bits of a value can be named, and unpacked to boolean arrays."""
    assert ExampleBitMask.flagged_bits(np.int16(1 + 8)) == ['NODATA', 'LOWSNR'], \
        'Set bits were not reported in bit order'
    assert ExampleBitMask.flagged_bits(np.int16(0)) == [], \
        'An unflagged value should report no bits'
    with pytest.raises(DC3BitMaskError, match='single integer'):
        ExampleBitMask.flagged_bits(np.zeros(3, dtype=int))

    value = np.array([0, 1, 2], dtype=np.int16)
    unpacked = ExampleBitMask.unpack(value, flag=['NODATA', 'CR'])
    assert len(unpacked) == 2, 'unpack did not return one array per requested bit'
    assert np.array_equal(unpacked[0], [False, True, False]), 'First unpacked bit is wrong'


def test_consolidate():
    """Several bits can be collapsed into one."""
    value = np.array([0, 1, 2, 0], dtype=np.int16)
    value = ExampleBitMask.consolidate(value, ['NODATA', 'CR'], 'LOWSNR')
    assert np.array_equal(
        ExampleBitMask.flagged(value, flag='LOWSNR'), [False, True, True, False]
    ), 'Consolidation did not set the target bit exactly where a source bit was set'


# ----------------------------------------------------------------------
# Headers
# ----------------------------------------------------------------------
def test_header_round_trip():
    """Bit definitions are written to and read back from a header."""
    hdr = ExampleBitMask.to_header()
    assert hdr['EXBIT0'] == 'NODATA', 'Bit name was not written under its numbered keyword'
    assert hdr.comments['EXBIT0'] == 'Pixel has no data', \
        'Bit description was not written as the keyword comment'
    assert ExampleBitMask.parse_header(hdr) == {
        0: 'NODATA', 1: 'CR', 2: 'OLDFLAG', 3: 'LOWSNR'
    }, 'Parsing the header did not recover the full written bit layout'
    ExampleBitMask.validate_header(hdr)


def test_retiring_a_bit_does_not_invalidate_existing_files():
    """
    A file written before a bit was retired still validates.

    This is the whole point of retiring rather than deleting: the older mask
    has that bit set, and must remain readable.  Omitting retired bits from the
    header record would make every such file fail validation -- turning the
    mechanism that preserves old files into one that rejects them.
    """
    class BeforeRetirement(BitMask):
        prefix = 'EXBIT'
        bits = {
            'NODATA': 'Pixel has no data',
            'CR': 'Pixel contaminated by a cosmic ray',
            'OLDFLAG': 'Pixel was rejected by a since-removed test',
            'LOWSNR': 'Pixel has low signal-to-noise',
        }

    # A header written while OLDFLAG was still live
    hdr = BeforeRetirement.to_header()
    # ... is still valid against the class that has since retired it
    ExampleBitMask.validate_header(hdr)


def test_long_descriptions_do_not_warn(recwarn):
    """
    An over-long description is truncated deliberately, not by astropy.

    astropy truncates it anyway, but with a VerifyWarning on every write.  The
    comment is a convenience; what the file needs to interpret a mask is the
    name-to-value mapping, which is the card's value and is never truncated.
    """
    class Verbose(BitMask):
        prefix = 'VRB'
        bits = {
            'AVERYLONGBITNAME': (
                'A description far longer than a FITS card can hold, which must therefore be '
                'truncated somewhere, and it may as well be here where it is intentional.'
            ),
        }

    hdr = Verbose.to_header()
    assert len(recwarn.list) == 0, \
        f'Writing a long description warned: {[str(w.message) for w in recwarn.list]}'
    assert hdr['VRB0'] == 'AVERYLONGBITNAME', \
        'The bit name, which is what the mapping needs, must never be truncated'
    assert hdr.comments['VRB0'].endswith('...'), \
        'An over-long description should be marked as truncated'
    Verbose.validate_header(hdr)


@pytest.mark.parametrize('namelen', [1, 4, 8, 12, 20, 30, 40])
@pytest.mark.parametrize('desclen', [1, 20, 40, 47, 48, 60, 120, 400])
def test_cards_fit_for_any_name_and_description_length(namelen, desclen, recwarn):
    """
    No combination of bit name and description overruns a FITS card.

    The truncation point depends on the length of the bit name, since the value
    field pushes the comment to the right, so it is swept rather than checked at
    one size.
    """
    class Sweep(BitMask):
        prefix = 'SW'
        bits = {'N' * namelen: 'D' * desclen}

    card = str(Sweep.to_header().cards[0])
    assert len(recwarn.list) == 0, \
        f'name={namelen}, descr={desclen} warned: {[str(w.message) for w in recwarn.list]}'
    assert len(card) <= 80, f'name={namelen}, descr={desclen} produced an {len(card)}-char card'


def test_validate_header_detects_renumbering():
    """
    A mask written under different definitions is rejected, not reinterpreted.

    This is the failure that motivates writing the definitions alongside the
    data: the integers are still perfectly readable, they just mean something
    else.
    """
    class Renumbered(BitMask):
        prefix = 'EXBIT'
        bits = {
            'NODATA': 'Pixel has no data',
            'LOWSNR': 'Pixel has low signal-to-noise',   # was bit 3, now bit 1
        }

    hdr = ExampleBitMask.to_header()
    with pytest.raises(DC3BitMaskError, match='do not match'):
        Renumbered.validate_header(hdr)


def test_validate_header_detects_missing_definitions():
    """A header with no bit definitions cannot be interpreted at all."""
    with pytest.raises(DC3BitMaskError, match='records no bit definitions'):
        ExampleBitMask.validate_header(fits.Header())


def test_header_survives_a_file(tmp_path):
    """The definitions survive an actual FITS write and read."""
    f = tmp_path / 'mask.fits'
    mask = ExampleMaskArray((4,))
    mask.turn_on('CR', select=np.array([True, False, True, False]))
    fits.PrimaryHDU(data=mask.mask, header=mask.to_header()).writeto(f)
    with fits.open(f) as hdu:
        ExampleBitMask.validate_header(hdu[0].header)
        restored = ExampleMaskArray(hdu[0].data)
        assert np.array_equal(restored.CR, mask.CR), \
            'Mask values did not survive a round trip through a FITS file'


# ----------------------------------------------------------------------
# BitMaskArray
# ----------------------------------------------------------------------
def test_array_attribute_access():
    """A bit is readable by name as a boolean array."""
    mask = ExampleMaskArray((5,))
    mask.turn_on('CR', select=np.array([True, False, True, False, False]))
    assert np.array_equal(mask.CR, [True, False, True, False, False]), \
        'Attribute access did not return the boolean array for the named bit'
    assert not np.any(mask.NODATA), 'An unset bit reported as set'


def test_array_attribute_access_rejects_unknown_names():
    """An unknown attribute is an AttributeError, not a silent empty result."""
    mask = ExampleMaskArray((5,))
    with pytest.raises(AttributeError):
        mask.NO_SUCH_BIT


def test_array_turn_on_everywhere():
    """
    Omitting the selection sets the bit everywhere.

    The selection defaults to Ellipsis rather than None, because numpy reads
    None as a new axis and would reshape the array instead of selecting it all.
    """
    mask = ExampleMaskArray((3,))
    mask.turn_on('CR')
    assert np.all(mask.CR), 'Omitting the selection did not set the bit everywhere'
    assert mask.shape == (3,), 'Setting a bit everywhere changed the array shape'

    mask.turn_off('CR')
    assert not np.any(mask.CR), 'Omitting the selection did not clear the bit everywhere'
    assert mask.shape == (3,), 'Clearing a bit everywhere changed the array shape'


def test_array_adopts_an_existing_array():
    """An existing integer array can be adopted, and must be integer."""
    values = np.array([0, 2, 3], dtype=np.int16)
    mask = ExampleMaskArray(values)
    assert np.array_equal(mask.CR, [False, True, True]), \
        'An adopted array was not interpreted with the declared bits'
    with pytest.raises(DC3BitMaskError, match='integer type'):
        ExampleMaskArray(np.zeros(3, dtype=float))


def test_array_requires_a_bitmask():
    """An array class that declares no bitmask has meaningless values."""
    class Undeclared(BitMaskArray):
        pass

    with pytest.raises(DC3CodingError, match='does not declare a bitmask'):
        Undeclared((3,))


def test_array_flagged_bits():
    """The bits set at one element can be named."""
    mask = ExampleMaskArray((3,))
    mask.turn_on(['NODATA', 'LOWSNR'], select=1)
    assert mask.flagged_bits(1) == ['NODATA', 'LOWSNR'], \
        'The bits set at the selected element were not reported'
    assert mask.flagged_bits(0) == [], 'An unflagged element reported bits'
