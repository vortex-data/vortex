// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;

/// Mask an array without reading buffers.
///
/// This trait is for mask implementations that can operate purely on array metadata and
/// structure without needing to read or execute on the underlying buffers. Implementations
/// should return `None` if masking requires buffer access.
///
/// The `mask` parameter is a boolean array where true=keep/valid, false=null-out.
///
/// # Preconditions
///
/// The mask is guaranteed to have the same length as the array. Trivial cases
/// (`AllValid`, `AllInvalid`, `NonNullable`) are handled by the caller before dispatch.
pub trait MaskReduce: VTable {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

/// Mask an array, potentially reading buffers.
///
/// Unlike [`MaskReduce`], this trait is for mask implementations that may need to read
/// and execute on the underlying buffers to produce the masked result.
///
/// The `mask` parameter is a boolean array where true=keep/valid, false=null-out.
///
/// # Preconditions
///
/// The mask is guaranteed to have the same length as the array. Trivial cases
/// (`AllValid`, `AllInvalid`, `NonNullable`) are handled by the caller before dispatch.
pub trait MaskKernel: VTable {
    fn mask(
        array: ArrayView<'_, Self>,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`MaskReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct MaskReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for MaskReduceAdaptor<V>
where
    V: MaskReduce,
{
    type Parent = ExactScalarFn<MaskExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, MaskExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only reduce the input child (index 0), not the mask child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        // Reduce only when the mask (child 1) is readable from metadata: a concrete `Bool` or a
        // `Constant`. `Mask::return_dtype` guarantees the mask is `Bool(NonNullable)`, so a
        // `Constant` here is a non-nullable Boolean. Other encodings may need execution, so leave
        // them to the kernel.
        let parent_ref: ArrayRef = (*parent).clone();
        let mask_child = parent_ref
            .nth_child(1)
            .ok_or_else(|| vortex_err!("Mask expression must have 2 children"))?;
        if mask_child.as_opt::<Bool>().is_none() && mask_child.as_opt::<Constant>().is_none() {
            return Ok(None);
        }
        <V as MaskReduce>::mask(array, &mask_child)
    }
}

/// Adaptor that wraps a [`MaskKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct MaskExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for MaskExecuteAdaptor<V>
where
    V: MaskKernel,
{
    type Parent = ExactScalarFn<MaskExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, MaskExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only execute the input child (index 0), not the mask child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let mask_child = parent
            .nth_child(1)
            .ok_or_else(|| vortex_err!("Mask expression must have 2 children"))?;
        <V as MaskKernel>::mask(array, &mask_child, ctx)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::executor::VortexSessionExecute;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::fns::mask::Mask as MaskExpr;

    /// A constant Boolean mask child must take the metadata-only reduction path (pushing into the
    /// input encoding) rather than surviving as a `ScalarFn` wrapper that falls through to
    /// execution. Asserting the optimized encoding makes this fail before the adaptor accepts
    /// `Constant` masks, not just verifying values that could pass through the execution fallback.
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn constant_mask_reduces_into_input(#[case] mask_value: bool) -> VortexResult<()> {
        let input = buffer![1i32, 2, 3, 4, 5].into_array();
        let mask = ConstantArray::new(
            Scalar::bool(mask_value, Nullability::NonNullable),
            input.len(),
        )
        .into_array();

        let masked = MaskExpr.try_new_array(input.len(), EmptyOptions, [input, mask])?;
        assert!(
            masked.is::<ScalarFn>(),
            "expected an un-optimized ScalarFn wrapper before optimization"
        );

        let optimized = masked.optimize()?;
        assert!(
            !optimized.is::<ScalarFn>(),
            "constant mask should not fall through to execution, got {}",
            optimized.encoding_id()
        );
        assert!(
            optimized.is::<Primitive>(),
            "constant mask should reduce into the Primitive input, got {}",
            optimized.encoding_id()
        );

        let mut ctx = crate::array_session().create_execution_ctx();
        let expected = if mask_value {
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4), Some(5)])
        } else {
            PrimitiveArray::from_option_iter([None::<i32>, None, None, None, None])
        };
        assert_arrays_eq!(optimized, expected, &mut ctx);

        Ok(())
    }
}
