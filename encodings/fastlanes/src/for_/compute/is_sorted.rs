// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRReferences;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;
use crate::unpack_iter::BitPacked as BitPackedIter;

#[derive(Debug)]
pub(crate) struct FoRIsSortedKernel;

impl DynAggregateKernel for FoRIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<FoR>() else {
            return Ok(None);
        };

        let result = match array.references() {
            FoRReferences::Global(_) => {
                let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
                let unsigned_array = PrimitiveArray::from_buffer_handle(
                    encoded.buffer_handle().clone(),
                    encoded.ptype().to_unsigned(),
                    encoded.validity()?,
                )
                .into_array();

                if options.strict {
                    is_strict_sorted(&unsigned_array, ctx)?
                } else {
                    is_sorted(&unsigned_array, ctx)?
                }
            }
            FoRReferences::Blocked(references) => {
                is_sorted_blocked(array, references, options.strict, ctx)?
            }
        };

        Ok(Some(IsSorted::make_partial(
            batch,
            result,
            options.strict,
            ctx,
        )?))
    }
}

fn is_sorted_blocked(
    array: ArrayView<'_, FoR>,
    references: &ArrayRef,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let len = array.len();
    if len <= 1 {
        return Ok(true);
    }

    let start_row = match array.validity()?.execute_mask(len, ctx)? {
        Mask::AllFalse(_) => return Ok(!strict),
        Mask::AllTrue(_) => 0usize,
        Mask::Values(mask_values) => {
            let null_count = len - mask_values.true_count();
            if strict && null_count > 1 {
                return Ok(false);
            }
            // In Vortex, nulls sort before all non-null values, so all nulls must form a prefix.
            if mask_values.bit_buffer().slice(0..null_count).true_count() != 0 {
                return Ok(false);
            }
            null_count
        }
    };

    if len - start_row <= 1 {
        return Ok(true);
    }

    let refs = references.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(refs.ptype(), |T| {
        is_sorted_blocked_t::<T>(array, refs.as_slice::<T>(), start_row, strict, ctx)
    })
}

fn is_sorted_blocked_t<T>(
    array: ArrayView<'_, FoR>,
    refs: &[T],
    start_row: usize,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>
where
    T: IntegerPType
        + PhysicalPType
        + BitPackedIter
        + PrimInt
        + WrappingAdd
        + AsPrimitive<T>
        + AsPrimitive<T::Physical>,
    T::Physical: BitPacking + NativePType + Zero + AsPrimitive<T>,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    if let Some(bp) = array.encoded().as_opt::<BitPacked>()
        && bp.offset() == array.offset()
    {
        if bp.dtype().as_ptype() == T::PTYPE {
            return is_sorted_blocked_bitpacked::<T, T>(bp, refs, start_row, strict, ctx);
        }
        return match_each_unsigned_integer_ptype!(bp.dtype().as_ptype(), |U| {
            is_sorted_blocked_bitpacked::<T, U>(bp, refs, start_row, strict, ctx)
        });
    }

    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = usize::from(array.offset());
    if encoded.ptype() == T::PTYPE {
        Ok(is_sorted_blocked_primitive::<T, T>(
            encoded.as_slice::<T>(),
            offset,
            refs,
            start_row,
            strict,
        ))
    } else {
        Ok(match_each_unsigned_integer_ptype!(encoded.ptype(), |U| {
            is_sorted_blocked_primitive::<T, U>(
                encoded.as_slice::<U>(),
                offset,
                refs,
                start_row,
                strict,
            )
        }))
    }
}

fn is_sorted_blocked_bitpacked<T, U>(
    bp: ArrayView<'_, BitPacked>,
    refs: &[T],
    start_row: usize,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>
where
    T: IntegerPType + PrimInt + WrappingAdd,
    U: PhysicalPType + BitPackedIter + AsPrimitive<U::Physical>,
    U::Physical: BitPacking + NativePType + Zero + AsPrimitive<T>,
{
    let len = bp.len();
    let offset = usize::from(bp.offset());
    let bit_width = usize::from(bp.bit_width());

    let patch_data = if let Some(patches) = bp.patches() {
        let p_idx = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let p_val = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        Some((p_idx, p_val, patches.offset()))
    } else {
        None
    };

    if let Some((p_idx, p_val, p_off)) = patch_data {
        match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
            Ok(is_sorted_blocked_bitpacked_inner::<T, U, I>(
                bp.packed_slice::<U::Physical>(),
                bit_width,
                offset,
                len,
                start_row,
                strict,
                refs,
                Some((p_idx.as_slice::<I>(), p_val.as_slice::<U>(), p_off)),
            ))
        })
    } else {
        Ok(is_sorted_blocked_bitpacked_inner::<T, U, u64>(
            bp.packed_slice::<U::Physical>(),
            bit_width,
            offset,
            len,
            start_row,
            strict,
            refs,
            None,
        ))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "typed blocked is_sorted execution parameters"
)]
fn is_sorted_blocked_bitpacked_inner<T, U, I>(
    packed: &[U::Physical],
    bit_width: usize,
    offset: usize,
    len: usize,
    start_row: usize,
    strict: bool,
    refs: &[T],
    patches: Option<(&[I], &[U], usize)>,
) -> bool
where
    T: IntegerPType + PrimInt + WrappingAdd,
    U: PhysicalPType + BitPackedIter + AsPrimitive<U::Physical>,
    U::Physical: BitPacking + NativePType + Zero + AsPrimitive<T>,
    I: AsPrimitive<usize>,
{
    let elems_per_chunk = 128 * bit_width / size_of::<U>();
    let padded_len = offset + len;
    let num_chunks = padded_len.div_ceil(FL_CHUNK_SIZE);
    let start_chunk = (offset + start_row) / FL_CHUNK_SIZE;
    let mut scratch = [U::Physical::zero(); FL_CHUNK_SIZE];
    let mut patch_cursor = patches
        .map(|(p_idx, _, p_off)| {
            p_idx.partition_point(|&idx| idx.as_().saturating_sub(p_off) < start_row)
        })
        .unwrap_or(0);
    let mut prev: Option<T> = None;

    for chunk in start_chunk..num_chunks {
        let ref_b = refs[chunk];
        let chunk_start_padded = chunk * FL_CHUNK_SIZE;
        let chunk_end_padded = (chunk_start_padded + FL_CHUNK_SIZE).min(padded_len);
        let chunk_lo = chunk_start_padded.saturating_sub(offset).max(start_row);
        let chunk_hi = chunk_end_padded - offset;
        let lane_start = offset + chunk_lo - chunk_start_padded;
        let lane_end = offset + chunk_hi - chunk_start_padded;

        if bit_width == 0 {
            scratch[lane_start..lane_end].fill(U::Physical::zero());
        } else {
            let packed_chunk = &packed[chunk * elems_per_chunk..][..elems_per_chunk];
            // SAFETY: `packed_chunk` holds one chunk at `bit_width` and `scratch` holds one chunk.
            unsafe { U::Physical::unchecked_unpack(bit_width, packed_chunk, &mut scratch) };
        }

        if let Some((p_idx, p_val, p_off)) = patches {
            while patch_cursor < p_idx.len() {
                let Some(pos) = p_idx[patch_cursor].as_().checked_sub(p_off) else {
                    patch_cursor += 1;
                    continue;
                };
                if pos >= chunk_hi {
                    break;
                }
                if pos >= chunk_lo {
                    scratch[offset + pos - chunk_start_padded] = p_val[patch_cursor].as_();
                }
                patch_cursor += 1;
            }
        }

        for &raw in &scratch[lane_start..lane_end] {
            let decoded = ref_b.wrapping_add(&raw.as_());
            if let Some(p) = prev {
                let ok = if strict { decoded > p } else { decoded >= p };
                if !ok {
                    return false;
                }
            }
            prev = Some(decoded);
        }
    }

    true
}

fn is_sorted_blocked_primitive<T, U>(
    encoded: &[U],
    offset: usize,
    refs: &[T],
    start_row: usize,
    strict: bool,
) -> bool
where
    T: NativePType + WrappingAdd + Ord,
    U: Copy + AsPrimitive<T>,
{
    let mut prev: Option<T> = None;
    let mut pos = start_row;
    let len = encoded.len();
    while pos < len {
        let abs_pos = offset + pos;
        let chunk = abs_pos / FL_CHUNK_SIZE;
        let run_len = (len - pos).min(FL_CHUNK_SIZE - (abs_pos % FL_CHUNK_SIZE));
        let ref_b = refs[chunk];
        for &val in &encoded[pos..pos + run_len] {
            let decoded = ref_b.wrapping_add(&val.as_());
            if let Some(p) = prev {
                let ok = if strict { decoded > p } else { decoded >= p };
                if !ok {
                    return false;
                }
            }
            prev = Some(decoded);
        }
        pos += run_len;
    }
    true
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
    use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedData;
    use crate::FoR;
    use crate::FoRData;
    use crate::for_::array::FoRArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_sorted() {
        let mut ctx = SESSION.create_execution_ctx();

        let a = PrimitiveArray::new(buffer![-1, 0, i8::MAX], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, i8::MAX], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, 30, 127], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, -3, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );

        let a = PrimitiveArray::new(buffer![-10, -3, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );

        let a = PrimitiveArray::new(buffer![-10, -11, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );

        let a = PrimitiveArray::new(buffer![-10, i8::MIN, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values(&ctx)
        );
    }

    #[test]
    fn test_blocked_is_sorted_and_strict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let strict_raw = PrimitiveArray::from_iter((0..3000i64).map(|i| i * 10 - 5000));
        let for_arr = FoR::encode_chunked(strict_raw, &mut ctx)?;
        let bp = BitPackedData::encode(for_arr.encoded(), 14, &mut ctx)?;
        let blocked = FoR::try_new_chunked(
            bp.into_array(),
            for_arr.blocked_references().unwrap().clone(),
            0,
        )?
        .into_array();
        assert!(is_sorted(&blocked, &mut ctx)?);
        assert!(is_strict_sorted(&blocked, &mut ctx)?);

        let sliced = blocked.slice(500..2500)?.optimize_ctx(&SESSION)?;
        assert!(is_sorted(&sliced, &mut ctx)?);
        assert!(is_strict_sorted(&sliced, &mut ctx)?);

        let mut dup_vec: Vec<i64> = (0..2500i64).collect();
        dup_vec[1024] = dup_vec[1023];
        let dup_raw = PrimitiveArray::from_iter(dup_vec);
        let dup_for = FoR::encode_chunked(dup_raw, &mut ctx)?;
        let dup_bp = BitPackedData::encode(dup_for.encoded(), 10, &mut ctx)?;
        let dup_blocked = FoR::try_new_chunked(
            dup_bp.into_array(),
            dup_for.blocked_references().unwrap().clone(),
            0,
        )?
        .into_array();
        assert!(is_sorted(&dup_blocked, &mut ctx)?);
        assert!(!is_strict_sorted(&dup_blocked, &mut ctx)?);

        let boundary_raw = PrimitiveArray::from_iter(
            (0..2500i64).map(|i| if i < 1024 { 1000 + i } else { 500 + (i - 1024) }),
        );
        let boundary_for = FoR::encode_chunked(boundary_raw, &mut ctx)?;
        let boundary_bp = BitPackedData::encode(boundary_for.encoded(), 10, &mut ctx)?;
        let boundary_blocked = FoR::try_new_chunked(
            boundary_bp.into_array(),
            boundary_for.blocked_references().unwrap().clone(),
            0,
        )?
        .into_array();
        assert!(!is_sorted(&boundary_blocked, &mut ctx)?);
        assert!(!is_strict_sorted(&boundary_blocked, &mut ctx)?);

        let leading_nulls =
            PrimitiveArray::from_option_iter((0..2500i32).map(|i| (i >= 2).then_some(i * 3)));
        let ln_for = FoR::encode_chunked(leading_nulls, &mut ctx)?.into_array();
        assert!(is_sorted(&ln_for, &mut ctx)?);
        assert!(!is_strict_sorted(&ln_for, &mut ctx)?);

        let one_null =
            PrimitiveArray::from_option_iter((0..2500i32).map(|i| (i >= 1).then_some(i * 3)));
        let on_for = FoR::encode_chunked(one_null, &mut ctx)?.into_array();
        assert!(is_sorted(&on_for, &mut ctx)?);
        assert!(is_strict_sorted(&on_for, &mut ctx)?);

        let interior_null =
            PrimitiveArray::from_option_iter((0..2500i32).map(|i| (i != 1500).then_some(i * 3)));
        let in_for = FoR::encode_chunked(interior_null, &mut ctx)?.into_array();
        assert!(!is_sorted(&in_for, &mut ctx)?);
        assert!(!is_strict_sorted(&in_for, &mut ctx)?);

        Ok(())
    }
}
