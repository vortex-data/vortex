// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingSub;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
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

/// FoR-specific is_constant kernel.
///
/// Delegates global-reference arrays to the encoded child and evaluates blocked-reference arrays
/// chunk-by-chunk with early exit on the first mismatch.
#[derive(Debug)]
pub(crate) struct FoRIsConstantKernel;

impl DynAggregateKernel for FoRIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<FoR>() else {
            return Ok(None);
        };

        let result = match array.references() {
            FoRReferences::Global(_) => is_constant(array.encoded(), ctx)?,
            FoRReferences::Blocked(references) => is_constant_blocked(array, references, ctx)?,
        };
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

fn is_constant_blocked(
    array: ArrayView<'_, FoR>,
    references: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let len = array.len();
    if len == 0 {
        return Ok(false);
    }
    if len == 1 {
        return Ok(true);
    }

    match array.validity()?.execute_mask(len, ctx)? {
        Mask::AllFalse(_) => return Ok(true),
        Mask::Values(_) => return Ok(false),
        Mask::AllTrue(_) => {}
    }

    let refs = references.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(refs.ptype(), |T| {
        is_constant_blocked_t::<T>(array, refs.as_slice::<T>(), ctx)
    })
}

fn is_constant_blocked_t<T>(
    array: ArrayView<'_, FoR>,
    refs: &[T],
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>
where
    T: IntegerPType
        + PhysicalPType
        + BitPackedIter
        + PrimInt
        + WrappingAdd
        + WrappingSub
        + AsPrimitive<T>
        + AsPrimitive<T::Physical>,
    T::Physical: BitPacking + NativePType + Zero + AsPrimitive<T> + AsPrimitive<u64>,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    if let Some(bp) = array.encoded().as_opt::<BitPacked>()
        && bp.offset() == array.offset()
    {
        if bp.dtype().as_ptype() == T::PTYPE {
            return is_constant_blocked_bitpacked::<T, T>(bp, refs, ctx);
        }
        return match_each_unsigned_integer_ptype!(bp.dtype().as_ptype(), |U| {
            is_constant_blocked_bitpacked::<T, U>(bp, refs, ctx)
        });
    }

    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = usize::from(array.offset());
    if encoded.ptype() == T::PTYPE {
        Ok(is_constant_blocked_primitive::<T, T>(
            encoded.as_slice::<T>(),
            offset,
            refs,
        ))
    } else {
        Ok(match_each_unsigned_integer_ptype!(encoded.ptype(), |U| {
            is_constant_blocked_primitive::<T, U>(encoded.as_slice::<U>(), offset, refs)
        }))
    }
}

fn is_constant_blocked_bitpacked<T, U>(
    bp: ArrayView<'_, BitPacked>,
    refs: &[T],
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>
where
    T: IntegerPType
        + PhysicalPType
        + PrimInt
        + WrappingAdd
        + WrappingSub
        + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<u64>,
    U: PhysicalPType + BitPackedIter + AsPrimitive<U::Physical>,
    U::Physical: BitPacking + NativePType + Zero + AsPrimitive<T>,
{
    let len = bp.len();
    let offset = usize::from(bp.offset());
    let bit_width = usize::from(bp.bit_width());
    let max_packed: u64 = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let patch_data = if let Some(patches) = bp.patches() {
        let p_idx = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let p_val = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        Some((p_idx, p_val, patches.offset()))
    } else {
        None
    };

    if let Some((p_idx, p_val, p_off)) = patch_data {
        match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
            Ok(is_constant_blocked_bitpacked_inner::<T, U, I>(
                bp.packed_slice::<U::Physical>(),
                bit_width,
                max_packed,
                offset,
                len,
                refs,
                Some((p_idx.as_slice::<I>(), p_val.as_slice::<U>(), p_off)),
            ))
        })
    } else {
        Ok(is_constant_blocked_bitpacked_inner::<T, U, u64>(
            bp.packed_slice::<U::Physical>(),
            bit_width,
            max_packed,
            offset,
            len,
            refs,
            None,
        ))
    }
}

fn is_constant_blocked_bitpacked_inner<T, U, I>(
    packed: &[U::Physical],
    bit_width: usize,
    max_packed: u64,
    offset: usize,
    len: usize,
    refs: &[T],
    patches: Option<(&[I], &[U], usize)>,
) -> bool
where
    T: IntegerPType
        + PhysicalPType
        + PrimInt
        + WrappingAdd
        + WrappingSub
        + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<u64>,
    U: PhysicalPType + BitPackedIter + AsPrimitive<U::Physical>,
    U::Physical: BitPacking + NativePType + Zero + AsPrimitive<T>,
    I: AsPrimitive<usize>,
{
    let elems_per_chunk = 128 * bit_width / size_of::<U>();
    let padded_len = offset + len;
    let num_chunks = padded_len.div_ceil(FL_CHUNK_SIZE);
    let mut scratch = [U::Physical::zero(); FL_CHUNK_SIZE];
    let mut patch_cursor = 0usize;
    let mut expected: Option<T> = None;

    for chunk in 0..num_chunks {
        let ref_b = refs[chunk];
        let chunk_start_padded = chunk * FL_CHUNK_SIZE;
        let chunk_end_padded = (chunk_start_padded + FL_CHUNK_SIZE).min(padded_len);
        let chunk_lo = chunk_start_padded.saturating_sub(offset);
        let chunk_hi = chunk_end_padded - offset;
        let lane_start = offset + chunk_lo - chunk_start_padded;
        let lane_end = offset + chunk_hi - chunk_start_padded;

        let chunk_has_patches = patches.is_some_and(|(p_idx, _, p_off)| {
            patch_cursor < p_idx.len()
                && p_idx[patch_cursor]
                    .as_()
                    .checked_sub(p_off)
                    .is_some_and(|pos| pos < chunk_hi)
        });

        if !chunk_has_patches && let Some(target) = expected {
            let needed: u64 =
                <T::Physical as AsPrimitive<u64>>::as_(target.wrapping_sub(&ref_b).as_());
            if needed > max_packed {
                return false;
            }
            if bit_width == 0 {
                // Every packed value in this unpatched chunk is 0, so it decodes to `ref_b`.
                if ref_b != target {
                    return false;
                }
                continue;
            }
        }

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

        let target =
            *expected.get_or_insert_with(|| ref_b.wrapping_add(&scratch[lane_start].as_()));
        for &raw in &scratch[lane_start..lane_end] {
            if ref_b.wrapping_add(&raw.as_()) != target {
                return false;
            }
        }
    }

    true
}

fn is_constant_blocked_primitive<T, U>(encoded: &[U], offset: usize, refs: &[T]) -> bool
where
    T: NativePType + WrappingAdd + Eq,
    U: Copy + AsPrimitive<T>,
{
    let first_len = (FL_CHUNK_SIZE - offset).min(encoded.len());
    let (first, rest) = encoded.split_at(first_len);
    let mut expected: Option<T> = None;
    for (chunk, &ref_b) in iter::once(first)
        .chain(rest.chunks(FL_CHUNK_SIZE))
        .zip(refs)
    {
        for &val in chunk {
            let decoded = ref_b.wrapping_add(&val.as_());
            let target = *expected.get_or_insert(decoded);
            if decoded != target {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_constant::is_constant;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedData;
    use crate::FoR;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::constant_across_different_refs(
        vec![100i64, 50, 0],
        vec![0u16; 3000],
        false,
    )]
    fn test_blocked_is_constant_varying(
        #[case] refs: Vec<i64>,
        #[case] base: Vec<u16>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = PrimitiveArray::from_iter(base).into_array();
        let bp = BitPackedData::encode(&encoded, 4, &mut ctx)?;
        let arr = FoR::try_new_chunked(bp.into_array(), Buffer::from(refs).into_array(), 0)?;
        assert_eq!(is_constant(&arr.into_array(), &mut ctx)?, expected);
        Ok(())
    }

    #[test]
    fn test_blocked_is_constant_true_with_different_chunk_refs() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Every element decodes to 1000i64 despite different chunk refs [1000, 990, 900].
        let residuals: Vec<u16> = (0..2500usize)
            .map(|i| match i / 1024 {
                0 => 0,
                1 => 10,
                _ => 100,
            })
            .collect();
        let refs = PrimitiveArray::from_iter([1000i64, 990, 900]).into_array();
        // Use bit_width = 4 so chunk 2 values (100) are stored as patches!
        let bp = BitPackedData::encode(
            &PrimitiveArray::from_iter(residuals).into_array(),
            4,
            &mut ctx,
        )?;
        let arr = FoR::try_new_chunked(bp.into_array(), refs, 0)?.into_array();
        assert!(is_constant(&arr, &mut ctx)?);

        // Non-zero offset slice across chunk boundary is also constant.
        let sliced = arr.slice(500..2200)?.optimize_ctx(&SESSION)?;
        assert!(is_constant(&sliced, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn test_blocked_is_constant_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let raw =
            PrimitiveArray::from_option_iter((0..2500i32).map(|i| (i != 1500).then_some(500)));
        // Construct blocked array directly with a single null in chunk 1.
        let residuals = PrimitiveArray::from_option_iter(
            (0..2500usize).map(|i| (i != 1500).then_some(if i < 1024 { 0u8 } else { 10u8 })),
        );
        let refs = PrimitiveArray::from_iter([500i32, 490, 490]).into_array();
        let arr = FoR::try_new_chunked(residuals.into_array(), refs, 0)?.into_array();
        assert_eq!(
            is_constant(&arr, &mut ctx)?,
            is_constant(&raw.into_array(), &mut ctx)?
        );
        Ok(())
    }
}
