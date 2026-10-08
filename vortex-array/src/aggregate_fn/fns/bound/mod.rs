// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared pieces of the byte-bounded extremum aggregates [`MaxBound`] and [`MinBound`].
//!
//! Both aggregates compute the extremum of a column and, for `Utf8`/`Binary` inputs, truncate it
//! to at most [`BoundOptions::max_bytes`] bytes so that a zone map over long strings stays small.
//! Truncation turns the exact extremum into a bound: a prefix is a lower bound of the minimum, and
//! an incremented prefix is an upper bound of the maximum. Each partial records whether its value
//! is the exact extremum or only a bound, so that readers can tell the two apart.
//!
//! Every partial is a nullable struct `{ value, is_exact }`:
//!
//! - A null struct means the input had no non-null values (`Empty`).
//! - `is_exact = true` means `value` is the exact extremum.
//! - `is_exact = false` means `value` is a bound of the extremum, or for [`MaxBound`] only, that
//!   `value` is null because no upper bound fits in `max_bytes` (`Unrepresentable`).
//!
//! These supersede the `vortex.bounded_max` and `vortex.bounded_min` aggregates, whose partials
//! neither record exactness nor share a representation; see [`MaxBound`] and [`MinBound`] for the
//! mapping from those partials.
//!
//! [`MaxBound`]: crate::aggregate_fn::fns::max_bound::MaxBound
//! [`MinBound`]: crate::aggregate_fn::fns::min_bound::MinBound

use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroUsize;
use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::scalar::Scalar;

/// Field name of the extremum, or its bound, in a [`MaxBound`] or [`MinBound`] partial.
///
/// [`MaxBound`]: crate::aggregate_fn::fns::max_bound::MaxBound
/// [`MinBound`]: crate::aggregate_fn::fns::min_bound::MinBound
pub const BOUND_VALUE: &str = "value";
/// Field name of the flag recording whether [`BOUND_VALUE`] is the exact extremum.
pub const BOUND_IS_EXACT: &str = "is_exact";

static NAMES: LazyLock<FieldNames> =
    LazyLock::new(|| FieldNames::from([BOUND_VALUE, BOUND_IS_EXACT]));

/// Options for the byte-bounded extremum aggregates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoundOptions {
    /// Maximum byte length of a `Utf8`/`Binary` value or bound. Fixed-width values are never
    /// truncated.
    pub max_bytes: NonZeroUsize,
}

impl BoundOptions {
    /// Bound `Utf8`/`Binary` values and bounds to `max_bytes` bytes.
    pub const fn new(max_bytes: NonZeroUsize) -> Self {
        Self { max_bytes }
    }

    /// Serialize the options as the little-endian `u64` byte limit.
    pub fn serialize(&self) -> VortexResult<Vec<u8>> {
        let max_bytes = u64::try_from(self.max_bytes.get())?;
        Ok(max_bytes.to_le_bytes().to_vec())
    }

    /// Deserialize options produced by [`BoundOptions::serialize`].
    pub fn deserialize(metadata: &[u8], name: &str) -> VortexResult<Self> {
        vortex_ensure_eq!(
            metadata.len(),
            size_of::<u64>(),
            "{name} options have the wrong byte length"
        );
        let mut bytes = [0u8; size_of::<u64>()];
        bytes.copy_from_slice(metadata);
        let max_bytes = usize::try_from(u64::from_le_bytes(bytes))?;
        vortex_ensure!(max_bytes > 0, "{name} requires max_bytes > 0");
        Ok(Self {
            max_bytes: NonZeroUsize::new(max_bytes).vortex_expect("checked non-zero max_bytes"),
        })
    }
}

impl Display for BoundOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.max_bytes.get())
    }
}

/// The input dtypes the bounded extremum aggregates accept: those with a minimum and maximum.
pub(crate) fn supported_dtype(input_dtype: &DType) -> Option<&DType> {
    MinMax
        .return_dtype(&NumericalAggregateOpts::default(), input_dtype)
        .map(|_| input_dtype)
}

/// The partial dtype `{ value: T, is_exact: bool }?` over `element_dtype`.
///
/// `value_nullability` is nullable for [`MaxBound`], whose `Unrepresentable` state has no value,
/// and non-nullable for [`MinBound`], which always has one.
///
/// [`MaxBound`]: crate::aggregate_fn::fns::max_bound::MaxBound
/// [`MinBound`]: crate::aggregate_fn::fns::min_bound::MinBound
pub fn bound_partial_dtype(element_dtype: &DType, value_nullability: Nullability) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                element_dtype.with_nullability(value_nullability),
                DType::Bool(Nullability::NonNullable),
            ],
        ),
        Nullability::Nullable,
    )
}

/// Build a non-null partial scalar of `partial_dtype` from its fields.
pub(crate) fn bound_partial_scalar(partial_dtype: &DType, value: Scalar, is_exact: bool) -> Scalar {
    Scalar::struct_(
        partial_dtype.clone(),
        vec![value, Scalar::bool(is_exact, Nullability::NonNullable)],
    )
}

/// Split a partial scalar into its `value` and `is_exact` fields, or `None` for a null partial.
pub(crate) fn parse_bound_partial(
    scalar: &Scalar,
    name: &str,
) -> VortexResult<Option<(Scalar, bool)>> {
    if scalar.is_null() {
        return Ok(None);
    }
    let Some(fields) = scalar.as_struct_opt() else {
        vortex_bail!("{name} partial must be a struct, got {}", scalar.dtype());
    };
    let Some(value) = fields.field_by_idx(0) else {
        vortex_bail!("{name} partial is missing its {BOUND_VALUE} field");
    };
    let Some(is_exact) = fields
        .field_by_idx(1)
        .and_then(|is_exact| is_exact.as_bool().value())
    else {
        vortex_bail!("{name} partial is missing its non-null {BOUND_IS_EXACT} field");
    };
    Ok(Some((value, is_exact)))
}

/// How a stored byte limit satisfies a requested one.
///
/// A bound computed under any byte limit is a sound bound of the extremum, so a differing limit
/// still satisfies the request approximately: a larger stored limit keeps more bytes and is
/// tighter, a smaller one is looser but never wrong.
pub(crate) fn byte_limit_satisfaction(
    stored: &BoundOptions,
    requested: &BoundOptions,
) -> AggregateFnSatisfaction {
    if stored == requested {
        AggregateFnSatisfaction::Exact
    } else {
        AggregateFnSatisfaction::Approximate
    }
}
