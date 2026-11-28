// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
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
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::BlockedChunkAction;
use super::for_each_decoded_blocked_chunk;
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
        + AsPrimitive<T::Physical>,
    T::Physical: BitPacking + NativePType + Zero + AsPrimitive<T> + AsPrimitive<u64>,
{
    if let Some(bp) = array.encoded().as_opt::<BitPacked>()
        && bp.offset() == array.offset()
    {
        let bit_width = usize::from(bp.bit_width());
        return is_constant_blocked_bitpacked::<T>(bp, bit_width, refs, ctx);
    }

    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = usize::from(array.offset());
    Ok(is_constant_blocked_primitive::<T>(
        encoded.as_slice::<T>(),
        offset,
        refs,
    ))
}

fn is_constant_blocked_bitpacked<T>(
    bp: ArrayView<'_, BitPacked>,
    bit_width: usize,
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
        + AsPrimitive<T::Physical>,
    T::Physical: BitPacking + NativePType + Zero + AsPrimitive<T> + AsPrimitive<u64>,
{
    let max_packed: u64 = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    let expected = Cell::new(None::<T>);

    for_each_decoded_blocked_chunk(
        bp,
        bit_width,
        refs,
        0,
        |ref_b, chunk_has_patches| {
            if !chunk_has_patches && let Some(target) = expected.get() {
                let needed: u64 =
                    <T::Physical as AsPrimitive<u64>>::as_(target.wrapping_sub(&ref_b).as_());
                if needed > max_packed {
                    return BlockedChunkAction::StopFalse;
                }
                if bit_width == 0 {
                    // Every packed value in this unpatched chunk is 0, so it decodes to `ref_b`.
                    return if ref_b != target {
                        BlockedChunkAction::StopFalse
                    } else {
                        BlockedChunkAction::Skip
                    };
                }
            }
            BlockedChunkAction::Unpack
        },
        |ref_b, lanes| {
            let target = match expected.get() {
                Some(t) => t,
                None => {
                    let t = ref_b.wrapping_add(&lanes[0].as_());
                    expected.set(Some(t));
                    t
                }
            };
            lanes
                .iter()
                .all(|&raw| ref_b.wrapping_add(&raw.as_()) == target)
        },
        ctx,
    )
}

fn is_constant_blocked_primitive<T>(encoded: &[T], offset: usize, refs: &[T]) -> bool
where
    T: NativePType + WrappingAdd + Eq,
{
    let first_len = (FL_CHUNK_SIZE - offset).min(encoded.len());
    let (first, rest) = encoded.split_at(first_len);
    let mut expected: Option<T> = None;
    for (chunk, &ref_b) in iter::once(first)
        .chain(rest.chunks(FL_CHUNK_SIZE))
        .zip(refs)
    {
        for val in chunk {
            let decoded = ref_b.wrapping_add(val);
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
        vec![0i64; 3000],
        false,
    )]
    fn test_blocked_is_constant_varying(
        #[case] refs: Vec<i64>,
        #[case] base: Vec<i64>,
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
        let residuals: Vec<i64> = (0..2500usize)
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
            (0..2500usize).map(|i| (i != 1500).then_some(if i < 1024 { 0i32 } else { 10i32 })),
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
