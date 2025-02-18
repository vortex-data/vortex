use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult};

use crate::array::ConstantArray;
use crate::compute::{binary_boolean, compare, BinaryOperator, Operator};
use crate::{Array, Canonical, Encoding, IntoArray};

pub trait BetweenFn<A> {
    fn between(
        &self,
        arr: &A,
        lower: &Array,
        lower_op: Operator,
        upper: &Array,
        upper_op: Operator,
    ) -> VortexResult<Option<Array>>;
}

impl<E: Encoding> BetweenFn<Array> for E
where
    E: BetweenFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn between(
        &self,
        arr: &Array,
        lower: &Array,
        lower_op: Operator,
        upper: &Array,
        upper_op: Operator,
    ) -> VortexResult<Option<Array>> {
        let (arr_ref, encoding) = arr.try_downcast_ref::<E>()?;
        BetweenFn::between(encoding, arr_ref, lower, lower_op, upper, upper_op)
    }
}

/// Compute the following expression, but will likely have a lower runtime
/// ```
///  use vortex_array::Array;
/// use vortex_array::compute::{binary_boolean, compare, BinaryOperator, Operator};///
/// use vortex_error::VortexResult;
///
/// fn between(
///    arr: impl AsRef<Array>,
///    lower: impl AsRef<Array>,
///    lower_op: Operator,
///    upper: impl AsRef<Array>,
///    upper_op: Operator) -> VortexResult<Array> {
///     binary_boolean(
///         &compare(lower, &arr, lower_op)?,
///         &compare(&arr, upper, upper_op)?,
///         BinaryOperator::And
///     )
/// }
///  ```
pub fn between(
    arr: impl AsRef<Array>,
    lower: impl AsRef<Array>,
    lower_op: Operator,
    upper: impl AsRef<Array>,
    upper_op: Operator,
) -> VortexResult<Array> {
    let arr = arr.as_ref();
    let lower = lower.as_ref();
    let upper = upper.as_ref();

    debug_assert!(arr.dtype().eq_ignore_nullability(lower.dtype()));
    debug_assert!(arr.dtype().eq_ignore_nullability(upper.dtype()));
    debug_assert_eq!(arr.len(), lower.len());
    debug_assert_eq!(arr.len(), upper.len());

    let result = between_impl(arr, lower, lower_op, upper, upper_op)?;

    debug_assert_eq!(result.len(), arr.len());
    debug_assert_eq!(
        result.dtype(),
        &DType::Bool(
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability()
        )
    );

    Ok(result)

    // println!("between {:?}", arr.encoding());
    // let arr = arr.clone().into_canonical()?.into_array();
    //
    // if let Some(result) = arr
    //     .vtable()
    //     .between_fn()
    //     .and_then(|f| f.between(&arr, lower, upper).transpose())
    //     .transpose()?
    // {
    //     return Ok(result);
    // }

    // todo!("between {:?}", arr.encoding())
}

fn between_impl(
    arr: impl AsRef<Array>,
    lower: impl AsRef<Array>,
    lower_op: Operator,
    upper: impl AsRef<Array>,
    upper_op: Operator,
) -> VortexResult<Array> {
    let arr = arr.as_ref();
    let lower = lower.as_ref();
    let upper = upper.as_ref();

    if let Some(lower) = ConstantArray::maybe_from(lower) {
        if lower.scalar().is_null() {
            return Ok(
                Canonical::empty(&arr.dtype().with_nullability(Nullability::Nullable)).into_array(),
            );
        }
    }

    if let Some(upper) = ConstantArray::maybe_from(upper) {
        if upper.scalar().is_null() {
            return Ok(
                Canonical::empty(&arr.dtype().with_nullability(Nullability::Nullable)).into_array(),
            );
        }
    }

    if let Some(result) = arr
        .vtable()
        .between_fn()
        .and_then(|f| f.between(arr, lower, lower_op, upper, upper_op).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    // TODO(joe): should we try to canonicalize the array and try between
    binary_boolean(
        &compare(lower, arr, lower_op)?,
        &compare(arr, upper, upper_op)?,
        BinaryOperator::And,
    )
}
