// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::builtins::ArrayBuiltins;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNull as FillNullExpr;
use crate::validity::Validity;

/// Fill nulls in an array with a scalar value without reading buffers.
///
/// This trait is for fill_null implementations that can operate purely on array metadata
/// and structure without needing to read or execute on the underlying buffers.
/// Implementations should return `None` if the operation requires buffer access.
///
/// # Preconditions
///
/// The fill value is guaranteed to be non-null. The array is guaranteed to have mixed
/// validity (neither all-valid nor all-invalid).
pub trait FillNullReduce: VTable {
    fn fill_null(array: ArrayView<'_, Self>, fill_value: &Scalar)
    -> VortexResult<Option<ArrayRef>>;
}

/// Fill nulls in an array with a scalar value, potentially reading buffers.
///
/// Unlike [`FillNullReduce`], this trait is for fill_null implementations that may need
/// to read and execute on the underlying buffers to produce the result.
///
/// # Preconditions
///
/// The fill value is guaranteed to be non-null. The array is guaranteed to have mixed
/// validity (neither all-valid nor all-invalid).
pub trait FillNullKernel: VTable {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Common preconditions for fill_null operations that apply to all arrays.
///
/// Returns `Some(result)` if the precondition short-circuits the fill_null operation,
/// or `None` if fill_null should proceed with the encoding-specific implementation.
pub(super) fn precondition(
    array: &ArrayRef,
    fill_value: &Scalar,
) -> VortexResult<Option<ArrayRef>> {
    vortex_ensure!(
        !fill_value.is_null(),
        "fill_null requires a non-null fill value"
    );

    // If the array has no nulls, fill_null is a no-op (just cast for nullability).
    if !array.dtype().is_nullable()
        || matches!(
            array.validity()?,
            Validity::NonNullable | Validity::AllValid
        )
    {
        return array.clone().cast(fill_value.dtype().clone()).map(Some);
    }

    // If all values are null, replace the entire array with the fill value.
    if array.validity()?.definitely_all_null() {
        return Ok(Some(
            ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        ));
    }

    Ok(None)
}

/// Fill null on a [`ConstantArray`] by replacing null scalars with the fill value,
/// or casting non-null scalars to the fill value's dtype.
pub(crate) fn fill_null_constant(
    array: ArrayView<Constant>,
    fill_value: &Scalar,
) -> VortexResult<ArrayRef> {
    let scalar = if array.scalar().is_null() {
        fill_value.clone()
    } else {
        array.scalar().cast(fill_value.dtype())?
    };
    Ok(ConstantArray::new(scalar, array.len()).into_array())
}

/// Adaptor that wraps a [`FillNullReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct FillNullReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for FillNullReduceAdaptor<V>
where
    V: FillNullReduce,
{
    type Parent = ExactScalarFn<FillNullExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, FillNullExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the input child (index 0), not the fill_value child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let fill_value = scalar_fn_array
            .get_child(1)
            .as_constant()
            .vortex_expect("fill_null fill_value must be constant");
        let arr = array.array().clone();
        if let Some(result) = precondition(&arr, &fill_value)? {
            return Ok(Some(result));
        }
        <V as FillNullReduce>::fill_null(array, &fill_value)
    }
}

/// Adaptor that wraps a [`FillNullKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct FillNullExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for FillNullExecuteAdaptor<V>
where
    V: FillNullKernel,
{
    type Parent = ExactScalarFn<FillNullExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, FillNullExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the input child (index 0), not the fill_value child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let fill_value = scalar_fn_array
            .get_child(1)
            .as_constant()
            .vortex_expect("fill_null fill_value must be constant");
        let arr = array.array().clone();
        if let Some(result) = precondition(&arr, &fill_value)? {
            return Ok(Some(result));
        }
        <V as FillNullKernel>::fill_null(array, &fill_value, ctx)
    }
}
