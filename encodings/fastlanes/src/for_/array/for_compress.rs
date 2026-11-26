// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use itertools::Itertools;
use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::stats::Stat;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferView;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::for_each_mask_word;
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

/// Subtracts `min` from every valid value. Null slots are written as zero so they cost no bits in
/// a downstream bit-packing, and the validity bit selects the zero without a branch per value.
fn encode_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    min: T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = parray.validity()?;
    let len = parray.len();
    let values = parray.as_slice::<T>();
    let subtract = |v: T| v.wrapping_sub(&min);

    let mut encoded = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let out = &mut encoded.spare_capacity_mut()[..len];
    match validity.execute_mask(len, ctx)? {
        Mask::AllTrue(_) => values.map_into(out, subtract),
        Mask::AllFalse(_) => out.fill(MaybeUninit::new(T::zero())),
        Mask::Values(mask) => values.map_masked_into(mask.bit_buffer(), out, subtract),
    }
    // SAFETY: each branch writes every lane of `out`, which spans exactly `len` items.
    unsafe { encoded.set_len(len) };

    Ok(PrimitiveArray::new(encoded.freeze(), validity))
}

fn encode_chunked_typed<T>(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<FoRArray>
where
    T: NativePType + WrappingSub + PrimInt + Default,
    PValue: From<T>,
{
    let validity = array.validity()?;
    let mask = validity.execute_mask(array.len(), ctx)?;
    let values = array.as_slice::<T>();

    let (references, bits_opt) = match mask.bit_buffer() {
        AllOr::All => (chunk_mins_all_valid(values), None),
        AllOr::Some(bits) => (chunk_mins_mixed_validity(values, bits), Some(bits)),
        // Every value is null, so constants stand in for both children.
        AllOr::None => {
            let dtype = array.dtype();
            return FoR::try_new(
                ConstantArray::new(Scalar::null(dtype.clone()), array.len()).into_array(),
                Scalar::zero_value(&dtype.as_nonnullable()),
            );
        }
    };

    // Every chunk has the same minimum, so one global reference serves them all.
    if references.iter().all_equal() {
        let reference = references.first().copied().unwrap_or_else(T::zero);
        let encoded = match bits_opt {
            None => subtract_all_valid(values, &references),
            Some(bits) => subtract_mixed_validity(values, bits, &references),
        };
        return FoR::try_new(
            PrimitiveArray::new(encoded, validity).into_array(),
            Scalar::primitive(reference, Nullability::NonNullable),
        );
    }

    let encoded = match bits_opt {
        None => subtract_all_valid(values, &references),
        Some(bits) => subtract_mixed_validity(values, bits, &references),
    };

    FoR::try_new_chunked(
        PrimitiveArray::new(encoded, validity).into_array(),
        PrimitiveArray::new(references, Validity::NonNullable).into_array(),
        0,
    )
}

/// Find each all-valid chunk's minimum.
fn chunk_mins_all_valid<T: PrimInt>(values: &[T]) -> Buffer<T> {
    values
        .chunks(FL_CHUNK_SIZE)
        .map(|chunk| chunk.iter().copied().min().unwrap_or_else(T::zero))
        .collect::<Buffer<T>>()
}

fn subtract_all_valid<T: PrimInt + WrappingSub>(values: &[T], references: &[T]) -> Buffer<T> {
    let mut encoded = BufferMut::<T>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    for ((chunk, out), &reference) in values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .zip(references)
    {
        for (slot, v) in out.iter_mut().zip(chunk) {
            slot.write(v.wrapping_sub(&reference));
        }
    }
    // SAFETY: the loop above initialized every value.
    unsafe { encoded.set_len(values.len()) };
    encoded.freeze()
}

/// Find each mixed-validity chunk's minimum valid value.
fn chunk_mins_mixed_validity<T: PrimInt>(values: &[T], bits: &BitBuffer) -> Buffer<T> {
    let bits = bits.as_view();
    let mins = values
        .chunks(FL_CHUNK_SIZE)
        .enumerate()
        .map(|(i, chunk)| {
            let start = i * FL_CHUNK_SIZE;
            let mask = bits.slice(start..start + chunk.len());
            valid_min(chunk, mask)
        })
        .collect::<Vec<_>>();

    // All-null chunks take the previous chunk's reference, or the first valid one at the start.
    let mut previous = mins
        .iter()
        .flatten()
        .next()
        .copied()
        .unwrap_or_else(T::zero);
    mins.into_iter()
        .map(|min| {
            previous = min.unwrap_or(previous);
            previous
        })
        .collect::<Buffer<T>>()
}

fn subtract_mixed_validity<T: PrimInt + WrappingSub + Default>(
    values: &[T],
    bits: &BitBuffer,
    references: &[T],
) -> Buffer<T> {
    let bits = bits.as_view();
    let mut encoded = BufferMut::<T>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    for ((i, (chunk, out)), &reference) in values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .enumerate()
        .zip(references)
    {
        let start = i * FL_CHUNK_SIZE;
        let mask = bits.slice(start..start + chunk.len());
        chunk.map_masked_into(mask, out, |v| v.wrapping_sub(&reference));
    }
    // SAFETY: the loop above initialized every value.
    unsafe { encoded.set_len(values.len()) };
    encoded.freeze()
}

/// The minimum of the valid `values`, or `None` if none are valid.
#[inline]
fn valid_min<T: PrimInt>(values: &[T], mask: BitBufferView) -> Option<T> {
    if mask.true_count() == 0 {
        return None;
    }
    let mut min = T::max_value();
    for_each_mask_word(mask, |word, start, len| {
        let bytes = word.to_le_bytes();
        for (j, &v) in values[start..start + len].iter().enumerate() {
            let valid = (bytes[j / 8] >> (j % 8)) & 1 == 1;
            // An `if` compiles to a conditional move. Blending with an all-ones mask instead took
            // 16 instructions per value and made encoding a third slower.
            min = min.min(if valid { v } else { T::max_value() });
        }
    });

    Some(min)
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use itertools::Itertools;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::primitive::PrimitiveArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::stats::StatsProvider;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::for_::array::FoRArraySlotsExt;
    use crate::for_::array::for_decompress::decompress;
    use crate::for_::tests::global;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_compress_round_trip_small() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new((1i32..10).collect::<Buffer<_>>(), Validity::NonNullable);
        let compressed = FoRData::encode(array.clone(), &mut ctx).unwrap();
        assert_eq!(i32::try_from(&global(&compressed).unwrap()).unwrap(), 1);

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
            u32::try_from(&global(&compressed).unwrap()).unwrap(),
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
        assert_eq!(global(&compressed).unwrap().dtype(), &dtype);
        assert!(global(&compressed).unwrap().dtype().is_signed_int());
        assert!(compressed.encoded().dtype().is_signed_int());

        let encoded = compressed.encoded().execute_scalar(0, &mut ctx).unwrap();
        assert_eq!(encoded, Scalar::from(0i32));
    }

    #[test]
    fn test_compress_nullable_zeroes_null_slots() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter(
            (0..200i32).map(|i| (i % 3 != 0).then_some(1_000 + i)),
        );
        let compressed = FoRData::encode(array.clone(), &mut ctx)?;
        assert_eq!(i32::try_from(&global(&compressed)?)?, 1_001);

        let encoded = compressed
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected: Vec<i32> = (0..200i32)
            .map(|i| if i % 3 != 0 { i - 1 } else { 0 })
            .collect();
        assert_eq!(encoded.as_slice::<i32>(), expected.as_slice());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_overflow() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = FoRData::encode(array.clone(), &mut ctx)?;
        assert_eq!(
            i8::MIN,
            global(&compressed)?
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

        let decompressed = decompress(compressed.clone(), &mut ctx)?;
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
