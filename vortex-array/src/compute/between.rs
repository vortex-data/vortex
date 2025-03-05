use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};

use crate::compute::{BinaryOperator, Operator, binary_boolean, compare};
use crate::{Array, ArrayRef, Canonical, Encoding, IntoArray};

pub trait BetweenFn<A> {
    fn between(
        &self,
        arr: A,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BetweenOptions {
    pub lower_strict: StrictComparison,
    pub upper_strict: StrictComparison,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StrictComparison {
    Strict,
    NonStrict,
}

impl StrictComparison {
    pub const fn to_operator(&self) -> Operator {
        match self {
            StrictComparison::Strict => Operator::Lt,
            StrictComparison::NonStrict => Operator::Lte,
        }
    }
}

impl<E: Encoding> BetweenFn<&dyn Array> for E
where
    E: for<'a> BetweenFn<&'a E::Array>,
{
    fn between(
        &self,
        arr: &dyn Array,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let array_ref = arr
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        BetweenFn::between(self, array_ref, lower, upper, options)
    }
}

/// Compute between (a <= x <= b), this can be implemented using compare and boolean and but this
/// will likely have a lower runtime.
///
/// This semantics is equivalent to:
/// ```
/// use vortex_array::{Array, ArrayRef};
/// use vortex_array::compute::{binary_boolean, compare, BetweenOptions, BinaryOperator, Operator};///
/// use vortex_error::VortexResult;
///
/// fn between(
///    arr: &dyn Array,
///    lower: &dyn Array,
///    upper: &dyn Array,
///    options: &BetweenOptions
/// ) -> VortexResult<ArrayRef> {
///     binary_boolean(
///         &compare(lower, arr, options.lower_strict.to_operator())?,
///         &compare(arr, upper,  options.upper_strict.to_operator())?,
///         BinaryOperator::And
///     )
/// }
///  ```
///
/// The BetweenOptions { lower: StrictComparison, upper: StrictComparison } defines if the
/// value is < (strict) or <= (non-strict).
///
pub fn between(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    debug_assert!(arr.dtype().eq_ignore_nullability(lower.dtype()));
    debug_assert!(arr.dtype().eq_ignore_nullability(upper.dtype()));
    debug_assert_eq!(arr.len(), lower.len());
    debug_assert_eq!(arr.len(), upper.len());

    let result = between_impl(arr, lower, upper, options)?;

    debug_assert_eq!(result.len(), arr.len());
    debug_assert_eq!(
        result.dtype(),
        &DType::Bool(
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability()
        )
    );

    Ok(result)
}

fn between_impl(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    if lower.as_constant().is_some_and(|v| v.is_null())
        || upper.as_constant().is_some_and(|v| v.is_null())
    {
        return Ok(
            Canonical::empty(&arr.dtype().with_nullability(Nullability::Nullable)).into_array(),
        );
    }

    if let Some(result) = arr
        .vtable()
        .between_fn()
        .and_then(|f| f.between(arr, lower, upper, options).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    // TODO(joe): should we try to canonicalize the array and try between
    binary_boolean(
        &compare(lower, arr, options.lower_strict.to_operator())?,
        &compare(arr, upper, options.upper_strict.to_operator())?,
        BinaryOperator::And,
    )
}
