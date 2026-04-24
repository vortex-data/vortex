// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::cast::Cast;

/// Reduce rule for cast: restructure the array without reading buffers.
///
/// Encodings implement this to push cast operations through their structure.
/// For example, RunEnd pushes cast down to its values array, ZigZag transforms
/// the target dtype to unsigned and pushes to its encoded array.
///
/// Returns `Ok(None)` if the rule doesn't apply to this array/dtype combination.
pub trait CastReduce: VTable {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>>;
}

/// Execute kernel for cast: perform the actual value conversion, potentially reading buffers.
///
/// Canonical array types implement this to do the real type conversion work.
/// For example, PrimitiveArray converts numeric values between types.
///
/// Returns `Ok(None)` if this kernel cannot handle the given dtype conversion.
pub trait CastKernel: VTable {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapts a [`CastReduce`] impl into an [`ArrayParentReduceRule`] for `ScalarFnArray(Cast, ...)`.
#[derive(Default, Debug)]
pub struct CastReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for CastReduceAdaptor<V>
where
    V: CastReduce,
{
    type Parent = ExactScalarFn<Cast>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Cast>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = parent.options;
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }
        <V as CastReduce>::cast(array, dtype)
    }
}

/// Adapts a [`CastKernel`] impl into an [`ExecuteParentKernel`] for `ScalarFnArray(Cast, ...)`.
#[derive(Default, Debug)]
pub struct CastExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for CastExecuteAdaptor<V>
where
    V: CastKernel,
{
    type Parent = ExactScalarFn<Cast>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = parent.options;
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }
        <V as CastKernel>::cast(array, dtype, ctx)
    }
}
