// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::vtable::EliasFano;
use super::vtable::EliasFanoArray;

/// Encode `array` as an [`EliasFano`] array, leaving both children uncompressed.
///
/// The children are left to the compressor: the EF scheme in `vortex-btrblocks` forces `upper`
/// into [`Unary`](crate::Unary) and bit-packs `lower`.
///
/// Returns `None` when the dtype is not a non-nullable unsigned integer, or when the values are
/// not sorted.
pub fn elias_fano_encode(array: ArrayView<'_, Primitive>) -> VortexResult<Option<EliasFanoArray>> {
    let &DType::Primitive(ptype, NonNullable) = array.dtype() else {
        return Ok(None);
    };
    if !ptype.is_unsigned_int() {
        return Ok(None);
    }

    match_each_unsigned_integer_ptype!(ptype, |T| {
        elias_fano_encode_typed(array.as_slice::<T>())
    })
}

/// [`elias_fano_encode`] for the values of one unsigned type.
fn elias_fano_encode_typed<T>(values: &[T]) -> VortexResult<Option<EliasFanoArray>>
where
    T: NativePType + PartialOrd + AsPrimitive<u64>,
    u64: AsPrimitive<T>,
{
    // Checked on the values, because a wrong `IsSorted` statistic would corrupt the output.
    if !values.is_sorted() {
        return Ok(None);
    }

    let max: u64 = values.last().map_or(0, |&max| max.as_());
    let lower_bits = elias_fano_lower_bits(max, values.len());
    let mask = (1u64 << lower_bits) - 1;

    let mut upper = BufferMut::<u64>::zeroed(usize::try_from(max >> lower_bits)? + 1);
    let mut lower = BufferMut::<T>::with_capacity(values.len());
    for &value in values {
        let value: u64 = value.as_();
        upper[AsPrimitive::<usize>::as_(value >> lower_bits)] += 1;
        lower.push((value & mask).as_());
    }

    EliasFano::try_new(
        PrimitiveArray::new(upper, Validity::NonNullable).into_array(),
        PrimitiveArray::new(lower, Validity::NonNullable).into_array(),
        u8::try_from(lower_bits)?,
    )
    .map(Some)
}

/// How many low bits of each value [`elias_fano_encode`] keeps, for `len` sorted values up to
/// `max`: `floor(log2(max / len))`, or 0 when `max < len`.
///
/// This is the split that makes the array smallest. Each low bit costs `len` bits in `lower`, and
/// a `Unary` `upper` costs one bit per value plus one per bucket. Once `max` reaches `len`, this
/// leaves between `len + 1` and `2 × len` buckets. One more low bit would cost `len` bits to save
/// at most `len` bucket bits, and one fewer would save `len` bits but add at least `len` buckets.
///
/// The split is the one in section 4 of [Quasi-Succinct Indices]. A `Unary` `upper` takes one bit
/// more than the paper's upper bits: the unset bit that closes the last bucket.
///
/// [Quasi-Succinct Indices]: https://arxiv.org/abs/1206.4300v1
pub fn elias_fano_lower_bits(max: u64, len: usize) -> u32 {
    (max / len.max(1) as u64).checked_ilog2().unwrap_or(0)
}
