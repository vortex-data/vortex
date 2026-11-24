// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
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
    T: PhysicalPType + WrappingSub + PrimInt + Default + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<u64> + AsPrimitive<u8> + AsPrimitive<u16> + AsPrimitive<u32>,
    PValue: From<T>,
{
    let validity = array.validity()?;
    let mask = validity.execute_mask(array.len(), ctx)?;
    let values = array.as_slice::<T>();

    let (references, max_span, bits_opt) = match mask.bit_buffer() {
        AllOr::All => {
            let (references, max_span) = chunk_mins_all_valid(values);
            (references, max_span, None)
        }
        AllOr::Some(bits) => {
            let (references, max_span) = chunk_mins_mixed_validity(values, bits);
            (references, max_span, Some(bits))
        }
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

    let encoded: ArrayRef = match narrowed_residual_ptype(T::PTYPE, max_span) {
        PType::U8 if T::PTYPE != PType::U8 => {
            let buf = match bits_opt {
                None => subtract_all_valid_narrow::<T, u8>(values, &references),
                Some(bits) => subtract_mixed_validity_narrow::<T, u8>(values, bits, &references),
            };
            PrimitiveArray::new(buf, validity).into_array()
        }
        PType::U16 if T::PTYPE != PType::U16 => {
            let buf = match bits_opt {
                None => subtract_all_valid_narrow::<T, u16>(values, &references),
                Some(bits) => subtract_mixed_validity_narrow::<T, u16>(values, bits, &references),
            };
            PrimitiveArray::new(buf, validity).into_array()
        }
        PType::U32 if T::PTYPE != PType::U32 => {
            let buf = match bits_opt {
                None => subtract_all_valid_narrow::<T, u32>(values, &references),
                Some(bits) => subtract_mixed_validity_narrow::<T, u32>(values, bits, &references),
            };
            PrimitiveArray::new(buf, validity).into_array()
        }
        _ => {
            let buf = match bits_opt {
                None => subtract_all_valid(values, &references),
                Some(bits) => subtract_mixed_validity(values, bits, &references),
            };
            PrimitiveArray::new(buf, validity).into_array()
        }
    };

    FoR::try_new_chunked(
        encoded,
        PrimitiveArray::new(references, Validity::NonNullable).into_array(),
        0,
    )
}

/// Choose the narrowest unsigned residual ptype for blocked FoR on 32-bit and 64-bit integers.
fn narrowed_residual_ptype(logical: PType, max_span: u64) -> PType {
    if logical.byte_width() < 4 {
        return logical;
    }
    if max_span <= u64::from(u8::MAX) {
        PType::U8
    } else if max_span <= u64::from(u16::MAX) {
        PType::U16
    } else if logical.byte_width() > 4 && max_span <= u64::from(u32::MAX) {
        PType::U32
    } else {
        logical
    }
}

/// Find each all-valid chunk's minimum and the maximum per-chunk span across all chunks.
fn chunk_mins_all_valid<T>(values: &[T]) -> (Buffer<T>, u64)
where
    T: PhysicalPType + PrimInt + WrappingSub + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<u64>,
{
    let mut max_span = 0u64;
    let references = values
        .chunks(FL_CHUNK_SIZE)
        .map(|chunk| {
            let mut min = T::max_value();
            let mut max = T::min_value();
            for &v in chunk {
                min = min.min(v);
                max = max.max(v);
            }
            let span: u64 = <T::Physical as AsPrimitive<u64>>::as_(max.wrapping_sub(&min).as_());
            max_span = max_span.max(span);
            min
        })
        .collect::<Buffer<T>>();
    (references, max_span)
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

fn subtract_all_valid_narrow<T, U>(values: &[T], references: &[T]) -> Buffer<U>
where
    T: PhysicalPType + PrimInt + WrappingSub + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<U>,
    U: NativePType,
{
    let mut encoded = BufferMut::<U>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    for ((chunk, out), &reference) in values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .zip(references)
    {
        for (slot, v) in out.iter_mut().zip(chunk) {
            slot.write(<T::Physical as AsPrimitive<U>>::as_(
                v.wrapping_sub(&reference).as_(),
            ));
        }
    }
    // SAFETY: the loop above initialized every value.
    unsafe { encoded.set_len(values.len()) };
    encoded.freeze()
}

/// Find each mixed-validity chunk's minimum valid value and the maximum per-chunk span.
fn chunk_mins_mixed_validity<T>(values: &[T], bits: &BitBuffer) -> (Buffer<T>, u64)
where
    T: PhysicalPType + PrimInt + WrappingSub + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<u64>,
{
    let bits = bits.as_view();
    let mut max_span = 0u64;
    let mins = values
        .chunks(FL_CHUNK_SIZE)
        .enumerate()
        .map(|(i, chunk)| {
            let start = i * FL_CHUNK_SIZE;
            let mask = bits.slice(start..start + chunk.len());
            valid_min_max(chunk, mask).map(|(min, max)| {
                let span: u64 =
                    <T::Physical as AsPrimitive<u64>>::as_(max.wrapping_sub(&min).as_());
                max_span = max_span.max(span);
                min
            })
        })
        .collect::<Vec<_>>();

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
    (references, max_span)
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

fn subtract_mixed_validity_narrow<T, U>(
    values: &[T],
    bits: &BitBuffer,
    references: &[T],
) -> Buffer<U>
where
    T: PhysicalPType + PrimInt + WrappingSub + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<U>,
    U: NativePType + Default,
{
    let bits = bits.as_view();
    let mut encoded = BufferMut::<U>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    for ((i, (chunk, out)), &reference) in values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .enumerate()
        .zip(references)
    {
        let start = i * FL_CHUNK_SIZE;
        let mask = bits.slice(start..start + chunk.len());
        chunk.map_masked_into(mask, out, |v| {
            <T::Physical as AsPrimitive<U>>::as_(v.wrapping_sub(&reference).as_())
        });
    }
    // SAFETY: the loop above initialized every value.
    unsafe { encoded.set_len(values.len()) };
    encoded.freeze()
}

/// The `(min, max)` of the valid `values`, or `None` if none are valid.
#[inline]
fn valid_min_max<T: PrimInt>(values: &[T], mask: BitBufferView) -> Option<(T, T)> {
    if mask.true_count() == 0 {
        return None;
    }
    let mut min = T::max_value();
    let mut max = T::min_value();
    for_each_mask_word(mask, |word, start, len| {
        let bytes = word.to_le_bytes();
        for (j, &v) in values[start..start + len].iter().enumerate() {
            let valid = (bytes[j / 8] >> (j % 8)) & 1 == 1;
            // An `if` compiles to a conditional move. Blending with an all-ones mask instead took
            // 16 instructions per value and made encoding a third slower.
            min = min.min(if valid { v } else { T::max_value() });
            max = max.max(if valid { v } else { T::min_value() });
        }
    });

    Some((min, max))
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
