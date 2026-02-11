// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;

use super::cast::cast;
use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExactScalarFn;
use crate::arrays::ScalarFnArrayView;
use crate::arrays::ScalarFnVTable;
use crate::builtins::ArrayBuiltins;
use crate::expr::FillNull as FillNullExpr;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::vtable::VTable;

/// Replace nulls in the array with another value.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::{PrimitiveArray};
/// use vortex_array::compute::{fill_null};
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let array = fill_null(array.as_ref(), &Scalar::from(42i32)).unwrap();
/// assert_eq!(array.display_values().to_string(), "[0i32, 42i32, 1i32, 42i32, 2i32]");
/// ```
// TODO(joe): deprecate me.
pub fn fill_null(array: &dyn Array, fill_value: &Scalar) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        !fill_value.is_null(),
        "fill_null requires a non-null fill value"
    );
    let result = array.to_array().fill_null(fill_value.clone())?;
    debug_assert!(
        fill_value.dtype().is_nullable() || !result.dtype().is_nullable(),
        "fill_null with non-nullable fill value must produce a non-nullable result"
    );
    Ok(result)
}

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
    fn fill_null(array: &Self::Array, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>>;
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
        array: &Self::Array,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Common preconditions for fill_null operations that apply to all arrays.
///
/// Returns `Some(result)` if the precondition short-circuits the fill_null operation,
/// or `None` if fill_null should proceed with the encoding-specific implementation.
fn precondition<V: VTable>(
    array: &V::Array,
    fill_value: &Scalar,
) -> VortexResult<Option<ArrayRef>> {
    // If the array has no nulls, fill_null is a no-op (just cast for nullability).
    if !array.dtype().is_nullable() || array.all_valid()? {
        return cast(&**array, fill_value.dtype()).map(Some);
    }

    // If all values are null, replace the entire array with the fill value.
    if array.all_invalid()? {
        return Ok(Some(
            ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        ));
    }

    Ok(None)
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
        array: &V::Array,
        parent: ScalarFnArrayView<'_, FillNullExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the input child (index 0), not the fill_value child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let fill_value = scalar_fn_array.children()[1]
            .as_constant()
            .vortex_expect("fill_null fill_value must be constant");
        if let Some(result) = precondition::<V>(array, &fill_value)? {
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
        array: &V::Array,
        parent: ScalarFnArrayView<'_, FillNullExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the input child (index 0), not the fill_value child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let fill_value = scalar_fn_array.children()[1]
            .as_constant()
            .vortex_expect("fill_null fill_value must be constant");
        if let Some(result) = precondition::<V>(array, &fill_value)? {
            return Ok(Some(result));
        }
        <V as FillNullKernel>::fill_null(array, &fill_value, ctx)
    }
}
