// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::builtins::ArrayBuiltins;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::vtable::ValidityHelper;

impl TakeExecute for Bool {
    fn take(
        array: &BoolArray,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices_nulls_zeroed = match indices.validity_mask()? {
            Mask::AllTrue(_) => indices.to_array(),
            Mask::AllFalse(_) => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                        .into_array(),
                ));
            }
            Mask::Values(_) => indices
                .to_array()
                .fill_null(Scalar::from(0).cast(indices.dtype())?)?,
        };
        let indices_nulls_zeroed = indices_nulls_zeroed.execute::<PrimitiveArray>(ctx)?;
        let buffer = match_each_integer_ptype!(indices_nulls_zeroed.ptype(), |I| {
            take_valid_indices(&array.to_bit_buffer(), indices_nulls_zeroed.as_slice::<I>())
        });

        Ok(Some(
            BoolArray::new(buffer, array.validity().take(indices)?).into_array(),
        ))
    }
}

/// Maximum number of u64 words we'll copy to the stack for the word-index fast path.
/// 64 words = 4096 bits, covering the common small-array case.
const MAX_STACK_WORDS: usize = 64;

fn take_valid_indices<I: AsPrimitive<usize>>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    if bools.len() <= MAX_STACK_WORDS * 64 {
        take_word_index(bools, indices)
    } else if indices.len() < bools.len() / 4 {
        take_no_lut(bools, indices)
    } else {
        take_lut_output_bytes(bools, indices)
    }
}

/// Fast path for small arrays: copy source bits into a stack-allocated `[u64]` array, then
/// gather bits directly into output bytes using word-level indexing.
///
/// This avoids both the O(n) heap LUT allocation of `take_lut_output_bytes` and the
/// per-bit closure overhead of `BitBuffer::collect_bool`. Each bit extraction is O(1)
/// via `words[idx / 64] >> (idx % 64) & 1` on cache-local stack data, and output is
/// packed 8 bits at a time with unrolled OR-shifts directly into output bytes.
fn take_word_index<I: AsPrimitive<usize>>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    let src_bytes = bools.inner().as_ref();
    let src_offset = bools.offset();
    let needed_words = bools.len().div_ceil(64);

    let mut words = [0u64; MAX_STACK_WORDS];
    for w in 0..needed_words {
        let byte_start = (src_offset + w * 64) / 8;
        let bit_off = (src_offset + w * 64) % 8;

        let bytes_avail = src_bytes.len() - byte_start;
        // SAFETY: byte_start is within src_bytes by BitBuffer invariants.
        let raw_word = if bytes_avail >= 8 {
            unsafe { (src_bytes.as_ptr().add(byte_start) as *const u64).read_unaligned() }
        } else {
            let mut tmp = 0u64;
            for b in 0..bytes_avail {
                tmp |= (unsafe { *src_bytes.get_unchecked(byte_start + b) } as u64) << (b * 8);
            }
            tmp
        };
        words[w] = raw_word >> bit_off;
    }

    pack_output_bytes(&words, indices, |words, idx| {
        ((words[idx / 64] >> (idx % 64)) & 1) as u8
    })
}

/// Zero-allocation path for sparse takes: extract bits directly from source bytes without
/// building a LUT. O(num_indices) with no upfront cost, but each bit extraction requires
/// a division + shift instead of a direct byte load.
///
/// Best when `indices.len()` is small relative to `src_len`, avoiding the wasted O(src_len)
/// LUT build that `take_lut_output_bytes` requires.
fn take_no_lut<I: AsPrimitive<usize>>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    let src_bytes = bools.inner().as_ref();
    let src_offset = bools.offset();

    pack_output_bytes(src_bytes, indices, |src, idx| {
        let abs = idx + src_offset;
        (src[abs / 8] >> (abs % 8)) & 1
    })
}

/// Dense-take path for large arrays: build a byte-per-bit LUT, then gather 8 bits at a time
/// directly into output bytes, skipping the intermediate u64 packing step.
///
/// Phase 1: Expand each source bit into a single byte (0 or 1) — O(src_len).
/// Phase 2: For each group of 8 output indices, gather from the LUT and OR-pack into one
///          output byte with explicit shifts — O(num_indices).
fn take_lut_output_bytes<I: AsPrimitive<usize>>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    let src_bytes = bools.inner().as_ref();
    let src_offset = bools.offset();
    let src_len = bools.len();

    let bit_lut = build_bit_lut_scalar(src_bytes, src_offset, src_len);

    pack_output_bytes(&bit_lut, indices, |lut, idx| lut[idx])
}

/// Build a byte-per-bit LUT: each source bit is expanded to a byte (0 or 1).
#[allow(clippy::uninit_vec)]
fn build_bit_lut_scalar(src_bytes: &[u8], src_offset: usize, src_len: usize) -> Vec<u8> {
    // SAFETY: every element in 0..src_len is written exactly once in the loop below.
    let mut bit_lut = Vec::with_capacity(src_len);
    unsafe { bit_lut.set_len(src_len) };
    for i in 0..src_len {
        let abs_bit = src_offset + i;
        // SAFETY: abs_bit / 8 is within src_bytes by BitBuffer invariants.
        bit_lut[i] = (unsafe { *src_bytes.get_unchecked(abs_bit / 8) } >> (abs_bit % 8)) & 1;
    }
    bit_lut
}

/// Pack indices into output bytes, 8 at a time, using a caller-provided bit extraction function.
fn pack_output_bytes<I: AsPrimitive<usize>, S: ?Sized, F>(
    src: &S,
    indices: &[I],
    get_bit: F,
) -> BitBuffer
where
    F: Fn(&S, usize) -> u8,
{
    let num_indices = indices.len();
    let num_bytes = num_indices.div_ceil(8);
    let mut out_bytes: Vec<u8> = vec![0u8; num_bytes];

    let full_bytes = num_indices / 8;
    for byte_idx in 0..full_bytes {
        let base = byte_idx * 8;
        // SAFETY: base + 7 < num_indices since byte_idx < full_bytes = num_indices / 8.
        // Each index value is within bounds by caller contract.
        unsafe {
            let mut byte_val = get_bit(src, indices.get_unchecked(base).as_());
            byte_val |= get_bit(src, indices.get_unchecked(base + 1).as_()) << 1;
            byte_val |= get_bit(src, indices.get_unchecked(base + 2).as_()) << 2;
            byte_val |= get_bit(src, indices.get_unchecked(base + 3).as_()) << 3;
            byte_val |= get_bit(src, indices.get_unchecked(base + 4).as_()) << 4;
            byte_val |= get_bit(src, indices.get_unchecked(base + 5).as_()) << 5;
            byte_val |= get_bit(src, indices.get_unchecked(base + 6).as_()) << 6;
            byte_val |= get_bit(src, indices.get_unchecked(base + 7).as_()) << 7;
            *out_bytes.get_unchecked_mut(byte_idx) = byte_val;
        }
    }

    if !num_indices.is_multiple_of(8) {
        let base = full_bytes * 8;
        let mut byte_val = 0u8;
        for bit in 0..(num_indices % 8) {
            let idx = unsafe { indices.get_unchecked(base + bit).as_() };
            byte_val |= get_bit(src, idx) << bit;
        }
        out_bytes[full_bytes] = byte_val;
    }

    BitBuffer::new(ByteBuffer::from(out_bytes), num_indices)
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::DynArray;
    use crate::IntoArray as _;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    #[test]
    fn take_nullable() {
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = reference
            .take(buffer![0, 3, 4].into_array())
            .unwrap()
            .to_bool();
        assert_eq!(
            b.to_bit_buffer(),
            BoolArray::from_iter([Some(false), None, Some(false)]).to_bit_buffer()
        );

        let all_invalid_indices = PrimitiveArray::from_option_iter([None::<i32>, None, None]);
        let b = reference.take(all_invalid_indices.into_array()).unwrap();
        assert_arrays_eq!(b, BoolArray::from_iter([None, None, None]));
    }

    #[test]
    fn test_bool_array_take_with_null_out_of_bounds_indices() {
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();

        // position 3 is null, the third index is null
        assert_arrays_eq!(actual, BoolArray::from_iter([Some(false), None, None]));
    }

    #[test]
    fn test_non_null_bool_array_take_with_null_out_of_bounds_indices() {
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        // the third index is null
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), Some(true), None])
        );
    }

    #[test]
    fn test_bool_array_take_all_null_indices() {
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_arrays_eq!(actual, BoolArray::from_iter([None, None, None]));
    }

    #[test]
    fn test_non_null_bool_array_take_all_null_indices() {
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_arrays_eq!(actual, BoolArray::from_iter([None, None, None]));
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true, false]))]
    #[case(BoolArray::from_iter([true]))]
    fn test_take_bool_conformance(#[case] array: BoolArray) {
        test_take_conformance(&array.into_array());
    }
}
