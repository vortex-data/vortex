// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::EliasFanoArray;
use super::EliasFanoArraySlotsExt;
use crate::Unary;

/// Decode every value. `lower` must be primitive, and `upper` must be [`Unary`], whose bits are
/// read directly, or primitive, whose counts are read bucket by bucket.
pub(super) fn elias_fano_decode(array: &EliasFanoArray) -> VortexResult<PrimitiveArray> {
    if let Some(upper) = array.upper().as_opt::<Unary>() {
        return decode_bits(array, &upper.to_bit_buffer());
    }

    let counts = array.upper().as_::<Primitive>();
    let counts = counts.as_slice::<u64>();

    // `validate` cannot see the children's values, so a malformed file is caught here.
    let total: u128 = counts.iter().map(|&count| u128::from(count)).sum();
    vortex_ensure_eq!(total, u128::try_from(array.len())?);

    let lower = array.lower().as_::<Primitive>();
    let lower_bits = usize::from(array.lower_bits);
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        elias_fano_decode_typed(counts, lower.as_slice::<T>(), lower_bits)
    })
}

/// Decode every value from the bits of a [`Unary`] upper. The value at `index` is the `index`-th
/// set bit, and the unset bits before it, one closing each earlier bucket, give its bucket.
fn decode_bits(array: &EliasFanoArray, bits: &BitBuffer) -> VortexResult<PrimitiveArray> {
    // Each set bit is one value, so a malformed file with too many or too few is caught here.
    vortex_ensure_eq!(bits.true_count(), array.len());

    let lower = array.lower().as_::<Primitive>();
    let lower_bits = usize::from(array.lower_bits);
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        decode_bits_typed(bits, lower.as_slice::<T>(), lower_bits)
    })
}

/// [`elias_fano_decode`] for the values of one unsigned type. The counts must sum to
/// `lows.len()`.
fn elias_fano_decode_typed<T: UnsignedPType>(
    counts: &[u64],
    lows: &[T],
    lower_bits: usize,
) -> VortexResult<PrimitiveArray>
where
    usize: AsPrimitive<T>,
{
    let mut lows = lows.iter();
    let mut values = BufferMut::<T>::with_capacity(lows.len());
    let mut stray = T::zero();

    for (bucket, &count) in counts.iter().enumerate() {
        let high = AsPrimitive::<T>::as_(bucket) << lower_bits;
        for &low in lows.by_ref().take(count.as_()) {
            stray = stray | low;
            values.push(high | low);
        }
    }

    vortex_ensure_eq!(
        stray >> lower_bits,
        T::zero(),
        "EliasFanoArray low bits must fit in lower_bits"
    );

    Ok(PrimitiveArray::new(values, Validity::NonNullable))
}

/// [`decode_bits`] for the values of one unsigned type. `bits` must have one set bit per value in
/// `lows`.
fn decode_bits_typed<T: UnsignedPType>(
    bits: &BitBuffer,
    lows: &[T],
    lower_bits: usize,
) -> VortexResult<PrimitiveArray>
where
    usize: AsPrimitive<T>,
{
    let mut values = BufferMut::<T>::with_capacity(lows.len());
    let mut stray = T::zero();

    bits.for_each_set_index(|bit| {
        let index = values.len();
        let low = lows[index];
        stray = stray | low;
        values.push((AsPrimitive::<T>::as_(bit - index) << lower_bits) | low);
    });

    vortex_ensure_eq!(
        stray >> lower_bits,
        T::zero(),
        "EliasFanoArray low bits must fit in lower_bits"
    );

    Ok(PrimitiveArray::new(values, Validity::NonNullable))
}
