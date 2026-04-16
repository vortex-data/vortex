// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod extension;
mod primitive;
mod varbin;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use self::bool::check_bool_sorted;
use self::decimal::check_decimal_sorted;
use self::extension::check_extension_sorted;
use self::primitive::check_primitive_sorted;
use self::varbin::check_varbinview_sorted;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::arrays::Constant;
use crate::arrays::Null;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;

/// Options for the `is_sorted` aggregate function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IsSortedOptions {
    /// If true, check for strictly ascending order (no duplicates).
    pub strict: bool,
}

impl Display for IsSortedOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "strict={}", self.strict)
    }
}

/// Compute whether an array is sorted in non-decreasing order.
///
/// Returns `true` for empty arrays and arrays of length 1.
/// Returns `false` for struct, list, and fixed-size list arrays.
pub fn is_sorted(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    is_sorted_impl(array, false, ctx)
}

/// Compute whether an array is strictly sorted in increasing order (no duplicates).
///
/// Returns `true` for empty arrays and arrays of length 1.
/// Returns `false` for struct, list, and fixed-size list arrays.
pub fn is_strict_sorted(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    is_sorted_impl(array, true, ctx)
}

fn is_sorted_impl(array: &ArrayRef, strict: bool, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let stat = if strict {
        Stat::IsStrictSorted
    } else {
        Stat::IsSorted
    };

    // Short-circuit using cached array statistics.
    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(stat) {
        return Ok(value);
    }

    // Arrays with 0 or 1 elements are (strict) sorted.
    if array.len() <= 1 {
        return Ok(true);
    }

    // Constant and null arrays are always sorted, but not strict sorted.
    if array.is::<Constant>() || array.is::<Null>() {
        let result = !strict;
        cache_is_sorted(array, strict, result);
        return Ok(result);
    }

    // We don't support sorting struct arrays.
    if array.dtype().is_struct() {
        return Ok(false);
    }

    // Short-circuit for unsupported dtypes.
    if IsSorted
        .return_dtype(&IsSortedOptions { strict }, array.dtype())
        .is_none()
    {
        return Ok(false);
    }

    // Enforce strictness before we even try to check if the array is sorted.
    if strict {
        let invalid_count = array.invalid_count(ctx)?;
        match invalid_count {
            // We can keep going
            0 => {}
            // If we have a potential null value - it has to be the first one.
            1 => {
                if !array.is_invalid(0, ctx)? {
                    cache_is_sorted(array, strict, false);
                    return Ok(false);
                }
            }
            _ => {
                cache_is_sorted(array, strict, false);
                return Ok(false);
            }
        }
    }

    // Compute using Accumulator<IsSorted>.
    let mut acc =
        Accumulator::try_new(IsSorted, IsSortedOptions { strict }, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result_scalar = acc.finish()?;

    let result = result_scalar.as_bool().value().unwrap_or(false);

    // Cache the computed result as statistics.
    cache_is_sorted(array, strict, result);

    Ok(result)
}

fn cache_is_sorted(array: &ArrayRef, strict: bool, result: bool) {
    let array_stats = array.statistics();
    if strict {
        if result {
            array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(true.into()));
        } else {
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
        }
    } else if result {
        array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
    } else {
        array_stats.set(Stat::IsSorted, Precision::Exact(false.into()));
        array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
    }
}

/// Aggregate function vtable for `is_sorted`.
///
/// Returns `Bool(NonNullable)` scalar.
/// The partial state is a nullable struct `{is_sorted: Bool(NN), first_value: T?, last_value: T?}`.
/// A null struct means the accumulator has seen no data yet (empty).
#[derive(Clone, Debug)]
pub struct IsSorted;

impl IsSorted {
    /// Build a partial scalar from a kernel's `is_sorted` result.
    ///
    /// Kernels that compute `is_sorted` by delegating to child arrays can call this
    /// to package the boolean result into the partial struct format expected by the
    /// accumulator, avoiding duplicated boilerplate.
    pub fn make_partial(
        batch: &ArrayRef,
        is_sorted: bool,
        strict: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let partial_dtype = make_is_sorted_partial_dtype(batch.dtype());
        if batch.is_empty() {
            return Ok(Scalar::null(partial_dtype));
        }
        let first_value = batch.execute_scalar(0, ctx)?.into_nullable();
        let last_value = batch.execute_scalar(batch.len() - 1, ctx)?.into_nullable();
        // SAFETY: We constructed partial_dtype and the children match its field dtypes exactly.
        Ok(unsafe {
            Scalar::struct_unchecked(
                partial_dtype,
                [
                    Scalar::bool(is_sorted, Nullability::NonNullable),
                    Scalar::bool(strict, Nullability::NonNullable),
                    first_value,
                    last_value,
                ],
            )
        })
    }
}

/// Partial accumulator state for is_sorted.
pub struct IsSortedPartial {
    is_sorted: bool,
    strict: bool,
    /// None = empty (no values seen).
    first_value: Option<Scalar>,
    last_value: Option<Scalar>,
    element_dtype: DType,
}

static NAMES: std::sync::LazyLock<FieldNames> = std::sync::LazyLock::new(|| {
    FieldNames::from(["is_sorted", "strict", "first_value", "last_value"])
});

pub fn make_is_sorted_partial_dtype(element_dtype: &DType) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                DType::Bool(Nullability::NonNullable),
                DType::Bool(Nullability::NonNullable),
                element_dtype.as_nullable(),
                element_dtype.as_nullable(),
            ],
        ),
        Nullability::Nullable,
    )
}

impl AggregateFnVTable for IsSorted {
    type Options = IsSortedOptions;
    type Partial = IsSortedPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.is_sorted")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("IsSorted is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Null
            | DType::Struct(..)
            | DType::List(..)
            | DType::FixedSizeList(..)
            | DType::Variant(..) => None,
            _ => Some(DType::Bool(Nullability::NonNullable)),
        }
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Null
            | DType::Struct(..)
            | DType::List(..)
            | DType::FixedSizeList(..)
            | DType::Variant(..) => None,
            _ => Some(make_is_sorted_partial_dtype(input_dtype)),
        }
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(IsSortedPartial {
            is_sorted: true,
            strict: options.strict,
            first_value: None,
            last_value: None,
            element_dtype: input_dtype.clone(),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if !partial.is_sorted {
            return Ok(());
        }

        // Null struct means the other accumulator was empty, skip it.
        if other.is_null() {
            return Ok(());
        }

        let other_is_sorted = other
            .as_struct()
            .field_by_idx(0)
            .map(|s| s.as_bool().value().unwrap_or(false))
            .unwrap_or(false);

        let other_first = other.as_struct().field_by_idx(2);
        let other_last = other.as_struct().field_by_idx(3);

        if !other_is_sorted {
            partial.is_sorted = false;
            // Still update last_value for correctness if needed, but we're done.
            if let Some(last) = other_last {
                partial.last_value = Some(last);
            }
            return Ok(());
        }

        // Check boundary: self.last_value vs other.first_value
        if let Some(ref self_last) = partial.last_value
            && let Some(ref other_first_val) = other_first
        {
            if !self_last.is_null() && !other_first_val.is_null() {
                let boundary_ok = if partial.strict {
                    *self_last < *other_first_val
                } else {
                    *self_last <= *other_first_val
                };
                if !boundary_ok {
                    partial.is_sorted = false;
                }
            } else if !self_last.is_null() && other_first_val.is_null() {
                // non-null before null violates sort order
                partial.is_sorted = false;
            } else if self_last.is_null() && other_first_val.is_null() && partial.strict {
                // both null with strict: violates strict sort
                partial.is_sorted = false;
            }
        }

        // Update first_value if this is the first non-empty chunk.
        if partial.first_value.is_none() {
            partial.first_value = other_first;
        }
        if let Some(last) = other_last {
            partial.last_value = Some(last);
        }

        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = make_is_sorted_partial_dtype(&partial.element_dtype);
        Ok(match (&partial.first_value, &partial.last_value) {
            (None, _) => {
                // Empty accumulator — return null struct.
                Scalar::null(dtype)
            }
            (Some(first_value), Some(last_value)) => {
                // SAFETY: We constructed partial_dtype and the children match its field dtypes.
                unsafe {
                    Scalar::struct_unchecked(
                        dtype,
                        [
                            Scalar::bool(partial.is_sorted, Nullability::NonNullable),
                            Scalar::bool(partial.strict, Nullability::NonNullable),
                            first_value.clone(),
                            last_value.clone(),
                        ],
                    )
                }
            }
            (Some(first_value), None) => {
                // SAFETY: We constructed partial_dtype and the children match its field dtypes.
                unsafe {
                    Scalar::struct_unchecked(
                        dtype,
                        [
                            Scalar::bool(partial.is_sorted, Nullability::NonNullable),
                            Scalar::bool(partial.strict, Nullability::NonNullable),
                            first_value.clone(),
                            first_value.clone(),
                        ],
                    )
                }
            }
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.is_sorted = true;
        partial.first_value = None;
        partial.last_value = None;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        !partial.is_sorted
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !partial.is_sorted {
            return Ok(());
        }

        match batch {
            Columnar::Constant(c) => {
                // Constant arrays are sorted but not strict sorted (if len > 1).
                let value = c.scalar().clone().into_nullable();
                if partial.strict && c.len() > 1 {
                    partial.is_sorted = false;
                }

                // Check boundary with previous chunk.
                if let Some(ref self_last) = partial.last_value {
                    if !self_last.is_null() && !value.is_null() {
                        let boundary_ok = if partial.strict {
                            *self_last < value
                        } else {
                            *self_last <= value
                        };
                        if !boundary_ok {
                            partial.is_sorted = false;
                        }
                    } else if (!self_last.is_null() && value.is_null())
                        || (self_last.is_null() && value.is_null() && partial.strict)
                    {
                        partial.is_sorted = false;
                    }
                }

                if partial.first_value.is_none() {
                    partial.first_value = Some(value.clone());
                }
                partial.last_value = Some(value);
                Ok(())
            }
            Columnar::Canonical(c) => {
                if c.is_empty() {
                    return Ok(());
                }

                let array_ref = c.clone().into_array();

                // Check boundary with previous chunk.
                let first_value = array_ref.execute_scalar(0, ctx)?.into_nullable();
                if let Some(ref self_last) = partial.last_value {
                    if !self_last.is_null() && !first_value.is_null() {
                        let boundary_ok = if partial.strict {
                            *self_last < first_value
                        } else {
                            *self_last <= first_value
                        };
                        if !boundary_ok {
                            partial.is_sorted = false;
                            partial.last_value = Some(
                                array_ref
                                    .execute_scalar(array_ref.len() - 1, ctx)?
                                    .into_nullable(),
                            );
                            if partial.first_value.is_none() {
                                partial.first_value = Some(first_value);
                            }
                            return Ok(());
                        }
                    } else if (!self_last.is_null() && first_value.is_null())
                        || (self_last.is_null() && first_value.is_null() && partial.strict)
                    {
                        partial.is_sorted = false;
                        partial.last_value = Some(
                            array_ref
                                .execute_scalar(array_ref.len() - 1, ctx)?
                                .into_nullable(),
                        );
                        if partial.first_value.is_none() {
                            partial.first_value = Some(first_value);
                        }
                        return Ok(());
                    }
                }

                // Check within-batch sortedness.
                let batch_is_sorted = match c {
                    Canonical::Primitive(p) => check_primitive_sorted(p, partial.strict, ctx)?,
                    Canonical::Bool(b) => check_bool_sorted(b, partial.strict, ctx)?,
                    Canonical::VarBinView(v) => check_varbinview_sorted(v, partial.strict)?,
                    Canonical::Decimal(d) => check_decimal_sorted(d, partial.strict, ctx)?,
                    Canonical::Extension(e) => check_extension_sorted(e, partial.strict, ctx)?,
                    Canonical::Null(_) => !partial.strict,
                    // Struct, List, FixedSizeList should have been filtered out by return_dtype
                    _ => unreachable!(),
                };

                if !batch_is_sorted {
                    partial.is_sorted = false;
                }

                if partial.first_value.is_none() {
                    partial.first_value = Some(first_value);
                }
                partial.last_value = Some(
                    array_ref
                        .execute_scalar(array_ref.len() - 1, ctx)?
                        .into_nullable(),
                );
                Ok(())
            }
        }
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        partials.get_item(NAMES.get(0).vortex_expect("out of bounds").clone())
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if partial.first_value.is_none() {
            // Empty accumulator → vacuously sorted.
            return Ok(Scalar::bool(true, Nullability::NonNullable));
        }
        Ok(Scalar::bool(partial.is_sorted, Nullability::NonNullable))
    }
}

#[expect(
    clippy::wrong_self_convention,
    reason = "is_* naming follows Iterator::is_sorted convention"
)]
/// Helper methods to check sortedness with strictness.
pub trait IsSortedIteratorExt: Iterator
where
    <Self as Iterator>::Item: PartialOrd,
{
    fn is_strict_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        self.is_sorted_by(|a, b| a < b)
    }
}

impl<T> IsSortedIteratorExt for T
where
    T: Iterator + ?Sized,
    T::Item: PartialOrd,
{
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::is_sorted::is_sorted;
    use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    // Tests migrated from compute/is_sorted.rs
    #[test]
    fn test_is_sorted() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let arr = PrimitiveArray::new(buffer!(0, 1, 2, 3), Validity::AllValid).into_array();
        assert!(is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(buffer!(0, 1, 3, 2), Validity::AllValid).into_array();
        assert!(!is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 3, 2),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_sorted(&arr, &mut ctx)?);

        Ok(())
    }

    #[test]
    fn test_is_strict_sorted() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let arr = PrimitiveArray::new(buffer!(0, 1, 2, 3), Validity::AllValid).into_array();
        assert!(is_strict_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(is_strict_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_strict_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 3, 2),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_strict_sorted(&arr, &mut ctx)?);

        Ok(())
    }

    // Tests migrated from arrays/primitive/compute/is_sorted.rs
    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2)]), true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1)]), true)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false)]
    fn test_primitive_is_sorted(#[case] array: PrimitiveArray, #[case] expected: bool) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            is_sorted(&array.into_array(), &mut ctx)
                .vortex_expect("operation should succeed in test"),
            expected
        );
    }

    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2), None]), false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1), None]), false)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false)]
    fn test_primitive_is_strict_sorted(#[case] array: PrimitiveArray, #[case] expected: bool) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            is_strict_sorted(&array.into_array(), &mut ctx)
                .vortex_expect("operation should succeed in test"),
            expected
        );
    }

    // Tests migrated from arrays/decimal/compute/is_sorted.rs
    #[test]
    fn test_decimal_is_sorted() -> VortexResult<()> {
        use arrow_array::types::Decimal128Type;
        use arrow_cast::parse::parse_decimal;

        use crate::arrays::DecimalArray;
        use crate::dtype::DecimalDType;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DecimalDType::new(19, 2);
        let i100 =
            parse_decimal::<Decimal128Type>("100.00", dtype.precision(), dtype.scale()).unwrap();
        let i200 =
            parse_decimal::<Decimal128Type>("200.00", dtype.precision(), dtype.scale()).unwrap();

        let sorted = buffer![i100, i200, i200];
        let unsorted = buffer![i200, i100, i200];

        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable);
        let unsorted_array = DecimalArray::new(unsorted, dtype, Validity::NonNullable);

        assert!(is_sorted(&sorted_array.into_array(), &mut ctx)?);
        assert!(!is_sorted(&unsorted_array.into_array(), &mut ctx)?);

        Ok(())
    }

    #[test]
    fn test_decimal_is_strict_sorted() -> VortexResult<()> {
        use arrow_array::types::Decimal128Type;
        use arrow_cast::parse::parse_decimal;

        use crate::arrays::DecimalArray;
        use crate::dtype::DecimalDType;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DecimalDType::new(19, 2);
        let i100 =
            parse_decimal::<Decimal128Type>("100.00", dtype.precision(), dtype.scale()).unwrap();
        let i200 =
            parse_decimal::<Decimal128Type>("200.00", dtype.precision(), dtype.scale()).unwrap();
        let i300 =
            parse_decimal::<Decimal128Type>("300.00", dtype.precision(), dtype.scale()).unwrap();

        let strict_sorted = buffer![i100, i200, i300];
        let sorted = buffer![i100, i200, i200];

        let dtype = DecimalDType::new(19, 2);

        let strict_sorted_array = DecimalArray::new(strict_sorted, dtype, Validity::NonNullable);
        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable);

        assert!(is_strict_sorted(
            &strict_sorted_array.into_array(),
            &mut ctx
        )?);
        assert!(!is_strict_sorted(&sorted_array.into_array(), &mut ctx)?);

        Ok(())
    }
}
