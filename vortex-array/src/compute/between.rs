use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult};

use crate::arrays::ConstantArray;
use crate::compute::{binary_boolean, compare, BinaryOperator, Operator};
use crate::{Array, Canonical, Encoding, IntoArray};

pub trait BetweenFn<A> {
    fn between(
        &self,
        arr: &A,
        lower: &Array,
        upper: &Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<Array>>;
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
    pub fn to_operator(&self) -> Operator {
        match self {
            StrictComparison::Strict => Operator::Lt,
            StrictComparison::NonStrict => Operator::Lte,
        }
    }
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
        upper: &Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<Array>> {
        let (arr_ref, encoding) = arr.try_downcast_ref::<E>()?;
        BetweenFn::between(encoding, arr_ref, lower, upper, options)
    }
}

/// Compute between (a <= x <= b), this can be implemented using compare and boolean andn but this
/// will likely have a lower runtime.
///
/// This semantics is equivalent to:
/// ```
/// use vortex_array::Array;
/// use vortex_array::compute::{binary_boolean, compare, BetweenOptions, BinaryOperator, Operator};///
/// use vortex_error::VortexResult;
///
/// fn between(
///    arr: impl AsRef<Array>,
///    lower: impl AsRef<Array>,
///    upper: impl AsRef<Array>,
///    options: &BetweenOptions
/// ) -> VortexResult<Array> {
///     binary_boolean(
///         &compare(lower, &arr, options.lower_strict.to_operator())?,
///         &compare(&arr, upper,  options.upper_strict.to_operator())?,
///         BinaryOperator::And
///     )
/// }
///  ```
///
/// The BetweenOptions { lower: StrictComparison, upper: StrictComparison } defines if the
/// value is < (strict) or <= (non-strict).
///
pub fn between(
    arr: impl AsRef<Array>,
    lower: impl AsRef<Array>,
    upper: impl AsRef<Array>,
    options: &BetweenOptions,
) -> VortexResult<Array> {
    let arr = arr.as_ref();
    let lower = lower.as_ref();
    let upper = upper.as_ref();

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
    arr: impl AsRef<Array>,
    lower: impl AsRef<Array>,
    upper: impl AsRef<Array>,
    options: &BetweenOptions,
) -> VortexResult<Array> {
    let arr = arr.as_ref();
    let lower = lower.as_ref();
    let upper = upper.as_ref();

    if ConstantArray::maybe_from(lower).is_some_and(|v| v.scalar().is_null())
        || ConstantArray::maybe_from(upper).is_some_and(|v| v.scalar().is_null())
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
