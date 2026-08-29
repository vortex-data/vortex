// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContains as ListContainsExpr;

fn constant_list_result(
    list: &ArrayRef,
    element_len: usize,
    element_nullability: crate::dtype::Nullability,
) -> Option<ArrayRef> {
    let list_scalar = list.as_constant()?;
    let DType::List(_, list_nullability) = list.dtype() else {
        return None;
    };
    let nullability = *list_nullability | element_nullability;

    match list_scalar.as_list().elements() {
        None => Some(
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), element_len).into_array(),
        ),
        Some(elements) if elements.is_empty() => {
            Some(ConstantArray::new(Scalar::bool(false, nullability), element_len).into_array())
        }
        Some(_) => None,
    }
}

/// Check list-contains without reading buffers (metadata-only).
///
/// This trait dispatches on the **element** (needle) child at index 1 of the `ListContains`
/// expression. `Self::Array` is the concrete element encoding, while the list (haystack) is
/// passed as an opaque `&ArrayRef`.
///
/// A future `ListContainsListReduce` could dispatch on the list side (child 0) for encodings
/// with specialized list representations.
///
/// The parent adaptor resolves null and empty constant lists before delegation.
///
/// Return `None` if the operation cannot be resolved from metadata alone.
pub trait ListContainsElementReduce: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Like [`ListContainsElementReduce`], this dispatches on the **element** (needle) child at
/// index 1. Unlike the reduce variant, implementations may read and execute on buffers via
/// the provided [`ExecutionCtx`].
///
/// The parent adaptor resolves null and empty constant lists before delegation.
pub trait ListContainsElementKernel: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`ListContainsElementReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ListContainsElementReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ListContainsElementReduceAdaptor<V>
where
    V: ListContainsElementReduce,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        if let Some(result) = constant_list_result(list, array.len(), array.dtype().nullability()) {
            return Ok(Some(result));
        }
        <V as ListContainsElementReduce>::list_contains(list, array)
    }
}

/// Adaptor that wraps a [`ListContainsElementKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ListContainsElementExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ListContainsElementExecuteAdaptor<V>
where
    V: ListContainsElementKernel,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        if let Some(result) = constant_list_result(list, array.len(), array.dtype().nullability()) {
            return Ok(Some(result));
        }
        <V as ListContainsElementKernel>::list_contains(list, array, ctx)
    }
}
