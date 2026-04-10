// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Add;

use num_traits::CheckedAdd;
use num_traits::CheckedSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::PValue;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_buffer::trusted_len::TrustedLen;
use vortex_error::VortexResult;

use crate::Sequence;
use crate::SequenceArray;
use crate::SequenceData;
/// An iterator that yields `base, base + step, base + 2*step, ...` via repeated addition.
struct SequenceIter<T> {
    acc: T,
    step: T,
    remaining: usize,
}

impl<T: Copy + Add<Output = T>> Iterator for SequenceIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        let val = self.acc;
        self.remaining -= 1;
        if self.remaining > 0 {
            self.acc = self.acc + self.step;
        }
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// SAFETY: `size_hint` returns an exact count and `next` yields exactly that many items.
unsafe impl<T: Copy + Add<Output = T>> TrustedLen for SequenceIter<T> {}

/// Decompresses a [`SequenceArray`] into a [`PrimitiveArray`].
#[inline]
pub fn sequence_decompress(array: &SequenceArray) -> VortexResult<ArrayRef> {
    fn decompress_inner<P: NativePType>(
        base: P,
        multiplier: P,
        len: usize,
        nullability: Nullability,
    ) -> PrimitiveArray {
        let values = BufferMut::from_trusted_len_iter(SequenceIter {
            acc: base,
            step: multiplier,
            remaining: len,
        });
        PrimitiveArray::new(values, Validity::from(nullability))
    }

    let prim = match_each_native_ptype!(array.ptype(), |P| {
        let base = array.base().cast::<P>()?;
        let multiplier = array.multiplier().cast::<P>()?;
        decompress_inner(base, multiplier, array.len(), array.dtype().nullability())
    });
    Ok(prim.into_array())
}

/// Encodes a primitive array into a sequence array, this is possible if:
/// 1. The array is not empty, and contains no nulls
/// 2. The array is not a float array. (This is due to precision issues, how it will stack well with ALP).
/// 3. The array is representable as a sequence `A[i] = base + i * multiplier` for multiplier != 0.
/// 4. The sequence has no deviations from the equation, this could be fixed with patches. However,
///    we might want a different array for that since sequence provide fast access.
pub fn sequence_encode(
    primitive_array: ArrayView<'_, Primitive>,
) -> VortexResult<Option<ArrayRef>> {
    if primitive_array.is_empty() {
        // we cannot encode an empty array
        return Ok(None);
    }

    if !primitive_array.array().all_valid()? {
        return Ok(None);
    }

    if primitive_array.ptype().is_float() {
        // for now, we don't handle float arrays, due to possible precision issues
        return Ok(None);
    }

    match_each_integer_ptype!(primitive_array.ptype(), |P| {
        encode_primitive_array(
            primitive_array.as_slice::<P>(),
            primitive_array.dtype().nullability(),
        )
    })
}

fn encode_primitive_array<P: NativePType + Into<PValue> + CheckedAdd + CheckedSub>(
    slice: &[P],
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    if slice.len() == 1 {
        // The multiplier here can be any value, zero is chosen
        return Sequence::try_new_typed(slice[0], P::zero(), nullability, 1)
            .map(|a| Some(a.into_array()));
    }
    let base = slice[0];
    let Some(multiplier) = slice[1].checked_sub(&base) else {
        return Ok(None);
    };

    if multiplier == P::zero() {
        return Ok(None);
    }

    if SequenceData::try_last(base.into(), multiplier.into(), P::PTYPE, slice.len()).is_err() {
        // If the last value is out of range, we cannot encode
        return Ok(None);
    }

    slice
        .windows(2)
        .all(|w| Some(w[1]) == w[0].checked_add(&multiplier))
        .then_some(
            Sequence::try_new_typed(base, multiplier, nullability, slice.len())
                .map(|a| a.into_array()),
        )
        .transpose()
}

#[cfg(test)]
mod tests {
    #[expect(unused_imports)]
    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use crate::sequence_encode;

    #[test]
    fn test_encode_array_success() {
        let primitive_array = PrimitiveArray::from_iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let encoded = sequence_encode(primitive_array.as_view()).unwrap();
        assert!(encoded.is_some());
        let decoded = encoded.unwrap().to_primitive();
        assert_arrays_eq!(decoded, primitive_array);
    }

    #[test]
    fn test_encode_array_1_success() {
        let primitive_array = PrimitiveArray::from_iter([0]);
        let encoded = sequence_encode(primitive_array.as_view()).unwrap();
        assert!(encoded.is_some());
        let decoded = encoded.unwrap().to_primitive();
        assert_arrays_eq!(decoded, primitive_array);
    }

    #[test]
    fn test_encode_array_fail() {
        let primitive_array = PrimitiveArray::from_iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);

        let encoded = sequence_encode(primitive_array.as_view()).unwrap();
        assert!(encoded.is_none());
    }

    #[test]
    fn test_encode_array_fail_oob() {
        let primitive_array = PrimitiveArray::from_iter(vec![100i8; 1000]);

        let encoded = sequence_encode(primitive_array.as_view()).unwrap();
        assert!(encoded.is_none());
    }

    #[test]
    fn test_encode_all_u8_values() {
        let primitive_array = PrimitiveArray::from_iter(0u8..=255);
        let encoded = sequence_encode(primitive_array.as_view()).unwrap();
        assert!(encoded.is_some());
        let decoded = encoded.unwrap().to_primitive();
        assert_arrays_eq!(decoded, primitive_array);
    }
}
