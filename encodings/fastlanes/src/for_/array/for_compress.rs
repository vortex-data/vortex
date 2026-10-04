// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRArray;
use crate::FoRData;

impl FoRData {
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<FoRArray> {
        let array_ref = array.clone().into_array();
        let min = array_ref
            .statistics()
            .compute_stat(Stat::Min, ctx)?
            .ok_or_else(|| vortex_err!("Min stat not found"))?;

        let encoded = match_each_integer_ptype!(array.ptype(), |T| {
            encode_primitive::<T>(array, T::try_from(&min)?, ctx)?.into_array()
        });
        FoR::try_new(encoded, min)
    }

    /// Encode with one reference per chunk: the minimum of the chunk's valid values.
    ///
    /// Chunks with no valid values reuse the previous chunk's reference, so the references
    /// compress into runs.
    pub fn encode_chunked(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<FoRArray> {
        match_each_integer_ptype!(array.ptype(), |T| {
            encode_chunked_typed::<T>(&array, ctx)
        })
    }
}

fn encode_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    min: T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    // Set null values to the min value, ensuring that decompress into a value in the primitive
    // range (and stop them wrapping around).
    let encoded = parray.map_each_with_validity::<T, _, _>(ctx, |(v, bool)| {
        if bool {
            v.wrapping_sub(&min)
        } else {
            T::zero()
        }
    })?;
    Ok(encoded)
}

fn encode_chunked_typed<T: NativePType + WrappingSub + PrimInt>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FoRArray>
where
    u8: AsPrimitive<T>,
{
    let validity = array.validity()?;
    let mask = validity.execute_mask(array.len(), ctx)?;
    let values = array.as_slice::<T>();
    let (encoded, references) = match mask.bit_buffer() {
        AllOr::All => encode_chunked_all_valid(values),
        AllOr::Some(bits) => encode_chunked_mixed_validity(values, bits),
        // Every value is null, so constants stand in for both children.
        AllOr::None => {
            let dtype = array.dtype();
            return FoR::try_new(
                ConstantArray::new(Scalar::null(dtype.clone()), array.len()).into_array(),
                Scalar::zero_value(&dtype.as_nonnullable()),
            );
        }
    };
    FoR::try_new_chunked(
        PrimitiveArray::new(encoded, validity).into_array(),
        PrimitiveArray::new(references, Validity::NonNullable).into_array(),
        0,
    )
}

/// Find each all-valid chunk's minimum and subtract it from each value while the chunk is in cache.
fn encode_chunked_all_valid<T: PrimInt + WrappingSub>(values: &[T]) -> (Buffer<T>, Buffer<T>) {
    let mut encoded = BufferMut::<T>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    let references = values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .map(|(chunk, out)| {
            let min = chunk.iter().copied().fold(T::max_value(), T::min);
            subtract(chunk, min, out);
            min
        })
        .collect::<Buffer<T>>();
    // SAFETY: the loop above initialized every value.
    unsafe { encoded.set_len(values.len()) };
    (encoded.freeze(), references)
}

fn subtract<T: PrimInt + WrappingSub>(values: &[T], reference: T, out: &mut [MaybeUninit<T>]) {
    for (out, v) in out.iter_mut().zip(values) {
        out.write(v.wrapping_sub(&reference));
    }
}

/// Find each mixed-validity chunk's minimum and subtract it from each non-null value while the chunk is in cache.
/// The minimum is the minimum non-null value.
fn encode_chunked_mixed_validity<T: PrimInt + WrappingSub + 'static>(
    values: &[T],
    bits: &BitBuffer,
) -> (Buffer<T>, Buffer<T>)
where
    u8: AsPrimitive<T>,
{
    // One validity bit per value, 64 values per word. `iter_padded` ends with the remainder word
    // even when it is empty, so keep one word per 64 values.
    let words: Vec<u64> = bits
        .chunks()
        .iter_padded()
        .take(values.len().div_ceil(64))
        .collect();

    let mut encoded = BufferMut::<T>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    let mins = values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .zip_eq(words.chunks(FL_CHUNK_SIZE / 64))
        .map(|((chunk, out), words)| {
            let min = valid_min(chunk, words);
            // An all-null chunk encodes as zeros whatever its reference.
            subtract_valid(chunk, words, min.unwrap_or_else(T::zero), out);
            min
        })
        .collect::<Vec<_>>();
    // SAFETY: the loop above initialized every value.
    unsafe { encoded.set_len(values.len()) };

    // All-null chunks take the previous chunk's reference, or the first valid one at the start.
    let mut previous = mins
        .iter()
        .flatten()
        .next()
        .copied()
        .unwrap_or_else(T::zero);
    let references = mins
        .into_iter()
        .map(|min| {
            previous = min.unwrap_or(previous);
            previous
        })
        .collect::<Buffer<T>>();
    (encoded.freeze(), references)
}

/// The minimum of the valid `values`, or `None` if none are valid.
#[inline]
fn valid_min<T: PrimInt + WrappingSub + 'static>(values: &[T], words: &[u64]) -> Option<T>
where
    u8: AsPrimitive<T>,
{
    if words.iter().all(|&word| word == 0) {
        return None;
    }
    let mut min = T::max_value();
    for_each_valid_mask(values, words, |_, v, mask| {
        min = min.min(select(mask, v, T::max_value()));
    });

    Some(min)
}

/// Subtract `reference` from the valid `values` and write zero for the invalid ones.
fn subtract_valid<T: PrimInt + WrappingSub + 'static>(
    values: &[T],
    words: &[u64],
    reference: T,
    out: &mut [MaybeUninit<T>],
) where
    u8: AsPrimitive<T>,
{
    for_each_valid_mask(values, words, |i, v, mask| {
        out[i].write(v.wrapping_sub(&reference) & mask);
    });
}

/// Calls `f(index, value, mask)` for each value, where `mask` is all ones for a valid value and
/// all zeros for a null.
///
/// Callers combine each value with its mask using bitwise operations instead of branching on
/// validity, which keeps their loops vectorized.
///
/// `words` holds one validity bit per value, least significant bit first, 64 values per word.
/// Each full 64-value block is a `[T; 64]` walked by a fixed `0..64` loop, so it unrolls and
/// vectorizes with no bounds checks. The remainder, fewer than 64 values, uses the word after the
/// full blocks and runs at most once per array.
///
/// Each bit is read from its byte of the word rather than by shifting the whole `u64`, which keeps
/// the vectorized loop in 8-bit lanes.
#[inline]
fn for_each_valid_mask<T: PrimInt + WrappingSub + 'static>(
    values: &[T],
    words: &[u64],
    mut f: impl FnMut(usize, T, T),
) where
    u8: AsPrimitive<T>,
{
    let (blocks, remainder) = values.as_chunks::<64>();
    for (block_idx, (block, &word)) in blocks.iter().zip(words).enumerate() {
        // Value `j`'s validity is bit `j % 8` of byte `j / 8`.
        let bytes = word.to_le_bytes();
        for j in 0..64 {
            // Shift the bit to the bottom and clear the rest: `1` if valid, `0` if null.
            let valid: T = ((bytes[j / 8] >> (j % 8)) & 1).as_();
            // Create all zero or one mask by subtracting from zero.
            f(block_idx * 64 + j, block[j], T::zero().wrapping_sub(&valid));
        }
    }
    // The remainder reads its bits the same way, from the word after the full blocks.
    let start = blocks.len() * 64;
    if let Some(&word) = words.get(blocks.len()) {
        let bytes = word.to_le_bytes();
        for (j, &v) in remainder.iter().enumerate() {
            let valid: T = ((bytes[j / 8] >> (j % 8)) & 1).as_();
            f(start + j, v, T::zero().wrapping_sub(&valid));
        }
    }
}

/// `a` where `mask` is all ones and `b` where it is all zeros.
#[inline]
fn select<T: PrimInt>(mask: T, a: T, b: T) -> T {
    (a & mask) | (b & !mask)
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use itertools::Itertools;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::primitive::PrimitiveArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::StatsProvider;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::for_::array::FoRArrayExt;
    use crate::for_::array::FoRArraySlotsExt;
    use crate::for_::array::for_decompress::decompress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_compress_round_trip_small() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new((1i32..10).collect::<Buffer<_>>(), Validity::NonNullable);
        let compressed = FoRData::encode(array.clone(), &mut ctx).unwrap();
        assert_eq!(
            i32::try_from(&compressed.constant_reference().unwrap()).unwrap(),
            1
        );

        assert_arrays_eq!(compressed, array, &mut ctx);
    }

    #[test]
    fn test_compress() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let array = PrimitiveArray::new(
            (0u32..10_000).map(|v| v + 1_000_000).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = FoRData::encode(array, &mut ctx).unwrap();
        assert_eq!(
            u32::try_from(&compressed.constant_reference().unwrap()).unwrap(),
            1_000_000u32
        );
    }

    #[test]
    fn test_zeros() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(buffer![0i32; 100], Validity::NonNullable);
        assert_eq!(array.statistics().len(), 0);

        let dtype = array.dtype().clone();
        let compressed = FoRData::encode(array, &mut ctx).unwrap();
        assert_eq!(compressed.constant_reference().unwrap().dtype(), &dtype);
        assert!(
            compressed
                .constant_reference()
                .unwrap()
                .dtype()
                .is_signed_int()
        );
        assert!(compressed.encoded().dtype().is_signed_int());

        let encoded = compressed.encoded().execute_scalar(0, &mut ctx).unwrap();
        assert_eq!(encoded, Scalar::from(0i32));
    }

    #[test]
    fn test_overflow() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = FoRData::encode(array.clone(), &mut ctx)?;
        assert_eq!(
            i8::MIN,
            compressed
                .constant_reference()
                .unwrap()
                .as_primitive()
                .typed_value::<i8>()
                .unwrap()
        );

        let encoded = compressed
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .reinterpret_cast(PType::U8);
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        let expected_unsigned = PrimitiveArray::from_iter(unsigned);
        assert_eq!(encoded.as_slice::<u8>(), expected_unsigned.as_slice::<u8>());

        let decompressed = decompress(&compressed, &mut ctx)?;
        array
            .as_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(
                    *v,
                    i8::try_from(&compressed.execute_scalar(i, &mut ctx).unwrap()).unwrap()
                );
            });
        assert_arrays_eq!(decompressed, array, &mut ctx);
        Ok(())
    }
}
