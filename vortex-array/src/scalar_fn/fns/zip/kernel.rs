// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::builtins::ArrayBuiltins;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::zip::Zip as ZipExpr;

/// Zip two arrays using a mask without reading buffers.
///
/// This trait is for zip implementations that can operate purely on array metadata
/// and structure without needing to read or execute on the underlying buffers.
/// Implementations should return `None` if the operation requires buffer access.
///
/// Dispatch supports either value branch. The adaptor normalizes the inputs so
/// `array` is selected where `mask` is true, and `if_false` everywhere else.
pub trait ZipReduce: VTable {
    fn zip(
        array: ArrayView<'_, Self>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Zip two arrays using a mask, potentially reading buffers.
///
/// Unlike [`ZipReduce`], this trait is for zip implementations that may need
/// to read and execute on the underlying buffers to produce the result.
///
/// Dispatch supports either value branch. The adaptor normalizes the inputs so
/// `array` is selected where `mask` is true, and `if_false` everywhere else.
pub trait ZipKernel: VTable {
    fn zip(
        array: ArrayView<'_, Self>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`ZipReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ZipReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ZipReduceAdaptor<V>
where
    V: ZipReduce,
{
    type Parent = ExactScalarFn<ZipExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ZipExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some((if_false, mask)) = normalized_zip_inputs(parent, child_idx)? else {
            return Ok(None);
        };
        <V as ZipReduce>::zip(array, &if_false, &mask)
    }
}

/// Adaptor that wraps a [`ZipKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ZipExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ZipExecuteAdaptor<V>
where
    V: ZipKernel,
{
    type Parent = ExactScalarFn<ZipExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ZipExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some((if_false, mask)) = normalized_zip_inputs(parent, child_idx)? else {
            return Ok(None);
        };
        <V as ZipKernel>::zip(array, &if_false, &mask, ctx)
    }
}

fn normalized_zip_inputs(
    parent: ScalarFnArrayView<'_, ZipExpr>,
    child_idx: usize,
) -> VortexResult<Option<(ArrayRef, ArrayRef)>> {
    let scalar_fn_array = parent
        .as_opt::<ScalarFn>()
        .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
    let mask = scalar_fn_array.get_child(2);
    match child_idx {
        0 => Ok(Some((scalar_fn_array.get_child(1).clone(), mask.clone()))),
        1 => {
            // Null mask rows select the original if_false branch, so they must
            // become true when that branch is passed to the kernel as if_true.
            let mask = mask.fill_null(false)?.not()?;
            Ok(Some((scalar_fn_array.get_child(0).clone(), mask)))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Bool;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;

    // A metadata-only reduction exposes the inputs passed by the adaptor.
    impl ZipReduce for Bool {
        fn zip(
            array: ArrayView<'_, Self>,
            if_false: &ArrayRef,
            mask: &ArrayRef,
        ) -> VortexResult<Option<ArrayRef>> {
            mask.zip(array.array().clone(), if_false.clone()).map(Some)
        }
    }

    #[rstest]
    #[case::if_true(0)]
    #[case::if_false(1)]
    #[case::mask(2)]
    fn adaptors_preserve_branch_and_null_mask_semantics(
        #[case] child_idx: usize,
    ) -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let children = [
            BoolArray::from_iter([Some(true), Some(false), None, Some(true)]).into_array(),
            BoolArray::from_iter([Some(false), Some(true), Some(true), None]).into_array(),
            BoolArray::from_iter([Some(true), Some(false), None, None]).into_array(),
        ];
        let parent = ZipExpr::try_new(
            children[0].clone(),
            children[1].clone(),
            children[2].clone(),
        )?
        .into_array();
        let child = children[child_idx].as_opt::<Bool>().unwrap();
        let reduced = ZipReduceAdaptor(Bool).reduce_parent(
            child,
            parent.as_opt::<ExactScalarFn<ZipExpr>>().unwrap(),
            child_idx,
        )?;
        let executed = ZipExecuteAdaptor(Bool).execute_parent(
            child,
            parent.as_opt::<ExactScalarFn<ZipExpr>>().unwrap(),
            child_idx,
            &mut ctx,
        )?;

        for result in [reduced, executed] {
            if child_idx == 2 {
                assert!(result.is_none(), "the mask is not a value branch");
            } else {
                let result = result.expect("both value branches dispatch to their kernel");
                let expected = BoolArray::from_iter([Some(true), Some(true), Some(true), None]);
                assert_arrays_eq!(result, expected, &mut ctx);
            }
        }
        Ok(())
    }
}
