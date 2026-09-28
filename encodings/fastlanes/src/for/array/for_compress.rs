// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

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
            compress_primitive::<T>(array, T::try_from(&min)?, ctx)?.into_array()
        });
        FoR::try_new(encoded, min)
    }

    /// Encode with one reference per chunk: the minimum of the chunk's valid values.
    ///
    /// Chunks with no valid values reuse the previous chunk's reference, so the references
    /// compress into runs.
    pub fn encode_chunked(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<FoRArray> {
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        let (encoded, references) = match_each_integer_ptype!(array.ptype(), |T| {
            let (encoded, references) = compress_chunked::<T>(array.as_slice::<T>(), &mask);
            (
                PrimitiveArray::new(encoded, array.validity()?),
                PrimitiveArray::new(references, Validity::NonNullable),
            )
        });
        FoR::try_new_chunked(encoded.into_array(), references.into_array(), 0)
    }
}

fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
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

fn compress_chunked<T: NativePType + WrappingSub + PrimInt>(
    values: &[T],
    mask: &Mask,
) -> (Buffer<T>, Buffer<T>)
where
    u8: AsPrimitive<T>,
{
    // One validity bit per value, 64 values per word. `None` means every value is valid.
    let words: Option<Vec<u64>> = match mask.bit_buffer() {
        AllOr::All => None,
        AllOr::None => Some(vec![0; values.len().div_ceil(64)]),
        AllOr::Some(bits) => Some(bits.chunks().iter_padded().collect()),
    };

    // Find each chunk's minimum and subtract it while the chunk is in cache. Null values don't
    // count towards the minimum and are set to zero, as in `compress_primitive`.
    let mut encoded = BufferMut::<T>::with_capacity(values.len());
    let mut mins = Vec::with_capacity(values.len().div_ceil(FL_CHUNK_SIZE));
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    for (chunk_idx, (chunk, out)) in values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .enumerate()
    {
        let min = match &words {
            None => {
                let min = chunk.iter().copied().fold(T::max_value(), T::min);
                subtract(chunk, min, out);
                Some(min)
            }
            Some(words) => {
                let words = &words[chunk_idx * (FL_CHUNK_SIZE / 64)..][..chunk.len().div_ceil(64)];
                let min = valid_min(chunk, words);
                // An all-null chunk encodes as zeros whatever its reference.
                subtract_valid(chunk, words, min.unwrap_or_else(T::zero), out);
                min
            }
        };
        mins.push(min);
    }
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

fn subtract<T: PrimInt + WrappingSub>(values: &[T], reference: T, out: &mut [MaybeUninit<T>]) {
    for (out, v) in out.iter_mut().zip(values) {
        out.write(v.wrapping_sub(&reference));
    }
}

/// The minimum of the valid `values`, or `None` if none are valid.
///
/// `words` holds one validity bit per value. Invalid values are replaced by `T::max_value()`
/// without branches, so the loop vectorizes.
fn valid_min<T: PrimInt + WrappingSub + 'static>(values: &[T], words: &[u64]) -> Option<T>
where
    u8: AsPrimitive<T>,
{
    let mut min = T::max_value();
    for_each_valid_mask(values, words, |_, v, mask| {
        min = min.min((v & mask) | (T::max_value() & !mask));
    });
    words.iter().any(|&word| word != 0).then_some(min)
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

/// Calls `f(index, value, mask)` for each value, where `mask` is all ones for valid values and
/// all zeros for invalid ones.
///
/// Full 64-value blocks run a fixed-length loop that reads each value's bit from its byte of the
/// validity word, so it vectorizes at every integer width.
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
        let bytes = word.to_le_bytes();
        for j in 0..64 {
            let valid: T = ((bytes[j / 8] >> (j % 8)) & 1).as_();
            f(block_idx * 64 + j, block[j], T::zero().wrapping_sub(&valid));
        }
    }
    let start = blocks.len() * 64;
    if let Some(&word) = words.get(blocks.len()) {
        for (j, &v) in remainder.iter().enumerate() {
            let valid: T = (((word >> j) & 1) as u8).as_();
            f(start + j, v, T::zero().wrapping_sub(&valid));
        }
    }
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
    use crate::BitPackedData;
    use crate::r#for::array::FoRArrayExt;
    use crate::r#for::array::FoRArraySlotsExt;
    use crate::r#for::array::for_decompress::decompress;
    use crate::r#for::array::for_decompress::fused_decompress;

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
    fn test_decompress() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = FoRData::encode(array.clone(), &mut ctx).unwrap();
        assert_arrays_eq!(compressed, array, &mut ctx);
    }

    #[test]
    fn test_decompress_fused() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 3, &mut ctx).unwrap();
        let compressed = FoR::try_new(bp.into_array(), 10u32.into()).unwrap();
        assert_arrays_eq!(compressed, expect, &mut ctx);
    }

    #[test]
    fn test_decompress_fused_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a range offset by a million.
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedData::encode(&array.into_array(), 2, &mut ctx)?;
        let compressed = FoR::try_new(bp.clone().into_array(), 10u32.into())?;
        let decompressed = fused_decompress::<u32>(&compressed, bp.as_view(), &mut ctx)?;
        assert_arrays_eq!(decompressed, expect, &mut ctx);
        Ok(())
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
