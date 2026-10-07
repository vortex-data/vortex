// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferView;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::for_each_masked_value;
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
    // A view slices each chunk's bits without cloning the buffer.
    let bits = bits.as_view();
    let mut encoded = BufferMut::<T>::with_capacity(values.len());
    let out = &mut encoded.spare_capacity_mut()[..values.len()];
    let mins = values
        .chunks(FL_CHUNK_SIZE)
        .zip(out.chunks_mut(FL_CHUNK_SIZE))
        .enumerate()
        .map(|(i, (chunk, out))| {
            let start = i * FL_CHUNK_SIZE;
            let mask = bits.slice(start..start + chunk.len());
            let min = valid_min(chunk, mask);
            // An all-null chunk encodes as zeros whatever its reference.
            subtract_valid(chunk, mask, min.unwrap_or_else(T::zero), out);
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
fn valid_min<T: PrimInt + WrappingSub + 'static>(values: &[T], mask: BitBufferView) -> Option<T>
where
    u8: AsPrimitive<T>,
{
    if mask.true_count() == 0 {
        return None;
    }
    let mut min = T::max_value();
    for_each_masked_value(values, mask, |_, v, valid| {
        min = min.min(select(lane_mask(valid), v, T::max_value()));
    });

    Some(min)
}

/// Subtract `reference` from the valid `values` and write zero for the invalid ones.
fn subtract_valid<T: PrimInt + WrappingSub + 'static>(
    values: &[T],
    mask: BitBufferView,
    reference: T,
    out: &mut [MaybeUninit<T>],
) where
    u8: AsPrimitive<T>,
{
    for_each_masked_value(values, mask, |i, v, valid| {
        out[i].write(v.wrapping_sub(&reference) & lane_mask(valid));
    });
}

/// All ones for a valid value and all zeros for a null, so callers combine each value with its
/// validity using bitwise operations instead of branching, which keeps their loops vectorized.
#[inline]
fn lane_mask<T: PrimInt + WrappingSub + 'static>(valid: bool) -> T
where
    u8: AsPrimitive<T>,
{
    T::zero().wrapping_sub(&u8::from(valid).as_())
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
    fn test_compress_nullable_zeroes_null_slots() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter(
            (0..200i32).map(|i| (i % 3 != 0).then_some(1_000 + i)),
        );
        let compressed = FoRData::encode(array.clone(), &mut ctx)?;
        let reference = compressed
            .constant_reference()
            .ok_or_else(|| vortex_err!("expected a constant reference"))?;
        assert_eq!(i32::try_from(&reference)?, 1_001);

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
