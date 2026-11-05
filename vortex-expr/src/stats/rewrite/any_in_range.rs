// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A pruning function that tests whether a zone's `[min, max]` range holds any value of a
//! sorted set.
//!
//! The falsifier for `needle IN (v1, ..., vn)` must show that no `vi` lies within the needle's
//! `[min, max]` bounds. As one comparison expression per element, it grows linearly with the list,
//! which is why large lists used to fall back to the list's overall bounds. [`AnyInRange`] instead
//! sorts the list once, when the pruning predicate is built, and binary searches it per zone.

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

/// `any_in_range(low, high)` is true for a row when some value of the options' sorted set lies in
/// `[low, high]`, and when either bound is null or NaN, since such a range may hold anything.
///
/// Both children must share the dtype of the set, ignoring nullability. The result is never null.
#[derive(Clone, Debug)]
pub(crate) struct AnyInRange;

/// The sorted, deduplicated, non-null and non-NaN values searched by [`AnyInRange`].
#[derive(Clone, Debug, PartialEq, Hash)]
pub(crate) struct AnyInRangeOptions {
    values: Arc<[Scalar]>,
}

// The values exclude NaN, the only scalars that are not equal to themselves.
impl Eq for AnyInRangeOptions {}

impl AnyInRangeOptions {
    /// Sorts and deduplicates the non-null, non-NaN `elements`.
    ///
    /// A null element or a NaN element never equals a value of the zone, provided the zone holds
    /// no NaN, so dropping them keeps the set's membership test sound under the NaN guards the
    /// falsifier adds. Returns `None` when no element remains or two elements are unordered.
    pub(crate) fn try_new(elements: &[Scalar]) -> Option<Self> {
        let mut values: Vec<Scalar> = elements
            .iter()
            .filter(|element| !element.is_null() && !is_nan(element))
            .cloned()
            .collect();
        if values.is_empty() {
            return None;
        }

        let mut unordered = false;
        values.sort_by(|left, right| {
            left.partial_cmp(right).unwrap_or_else(|| {
                unordered = true;
                Ordering::Equal
            })
        });
        if unordered {
            return None;
        }
        values.dedup();

        Some(Self {
            values: values.into(),
        })
    }

    /// The sorted values, at least one.
    pub(crate) fn values(&self) -> &[Scalar] {
        &self.values
    }

    fn dtype(&self) -> &DType {
        self.values[0].dtype()
    }
}

impl Display for AnyInRangeOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let first = &self.values[0];
        let last = &self.values[self.values.len() - 1];
        write!(f, "{} values in [{first}, {last}]", self.values.len())
    }
}

fn is_nan(scalar: &Scalar) -> bool {
    scalar
        .as_primitive_opt()
        .is_some_and(|primitive| primitive.is_nan())
}

impl ScalarFnVTable for AnyInRange {
    type Options = AnyInRangeOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.stats.any_in_range");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("low"),
            1 => ChildName::from("high"),
            _ => unreachable!("any_in_range has exactly two children"),
        }
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        for arg in args {
            vortex_ensure!(
                arg.eq_ignore_nullability(options.dtype()),
                "any_in_range bound {arg} does not match the value dtype {}",
                options.dtype()
            );
        }
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = args.row_count();
        let low = args.get(0)?;
        let high = args.get(1)?;
        let low_valid = low.validity()?.execute_mask(len, ctx)?;
        let high_valid = high.validity()?.execute_mask(len, ctx)?;
        let known = &low_valid & &high_valid;
        let values = options.values();

        let bits = match low.dtype() {
            DType::Primitive(ptype, _) => match_each_native_ptype!(ptype, |T| {
                let values = values
                    .iter()
                    .map(|value| {
                        value
                            .as_primitive()
                            .typed_value::<T>()
                            .ok_or_else(|| vortex_err!("any_in_range values must be non-null"))
                    })
                    .collect::<VortexResult<Vec<T>>>()?;
                let low = low.execute::<PrimitiveArray>(ctx)?;
                let high = high.execute::<PrimitiveArray>(ctx)?;
                let (low, high) = (low.as_slice::<T>(), high.as_slice::<T>());
                BitBuffer::collect_bool(len, |idx| {
                    let (low, high) = (low[idx], high[idx]);
                    !known.value(idx)
                        || low.is_nan()
                        || high.is_nan()
                        || intersects(&values, &low, &high)
                })
            }),
            DType::Utf8(_) | DType::Binary(_) => {
                let values = values
                    .iter()
                    .map(|value| {
                        value
                            .as_utf8_opt()
                            .and_then(|utf8| utf8.value().map(|value| value.as_bytes()))
                            .or_else(|| value.as_binary().value().map(|value| value.as_slice()))
                            .ok_or_else(|| vortex_err!("any_in_range values must be non-null"))
                    })
                    .collect::<VortexResult<Vec<&[u8]>>>()?;
                let low = low.execute::<VarBinViewArray>(ctx)?;
                let high = high.execute::<VarBinViewArray>(ctx)?;
                let (low, high) = (low.resolved_views(), high.resolved_views());
                BitBuffer::collect_bool(len, |idx| {
                    !known.value(idx) || intersects(&values, &low.bytes(idx), &high.bytes(idx))
                })
            }
            // Other orderable dtypes, such as decimals and extension types, compare as scalars.
            // Stats tables hold a row per zone, so this is a handful of comparisons per zone.
            _ => {
                let low = low.execute::<Canonical>(ctx)?.into_array();
                let high = high.execute::<Canonical>(ctx)?.into_array();
                let mut bits = Vec::with_capacity(len);
                for idx in 0..len {
                    bits.push(
                        !known.value(idx)
                            || scalars_intersect(
                                values,
                                &low.execute_scalar(idx, ctx)?,
                                &high.execute_scalar(idx, ctx)?,
                            ),
                    );
                }
                BitBuffer::from(bits)
            }
        };

        Ok(BoolArray::new(bits, Validity::NonNullable).into_array())
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Whether some value of the ascending `values` lies in `[low, high]`.
///
/// The bounds must be ordered with respect to every value.
fn intersects<T: PartialOrd>(values: &[T], low: &T, high: &T) -> bool {
    let first_at_least_low = values.partition_point(|value| value < low);
    values
        .get(first_at_least_low)
        .is_some_and(|value| value <= high)
}

/// [`intersects`] for scalars, which keeps the range when a bound is unordered with the values.
fn scalars_intersect(values: &[Scalar], low: &Scalar, high: &Scalar) -> bool {
    if low.partial_cmp(&values[0]).is_none() || high.partial_cmp(&values[0]).is_none() {
        return true;
    }
    intersects(values, low, high)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::VecExecutionArgs;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::AnyInRange;
    use super::AnyInRangeOptions;

    fn execute(values: &[Scalar], low: ArrayRef, high: ArrayRef) -> VortexResult<ArrayRef> {
        let options = AnyInRangeOptions::try_new(values).expect("values should be ordered");
        let len = low.len();
        let mut ctx = VortexSession::empty().create_execution_ctx();
        AnyInRange.execute(
            &options,
            &VecExecutionArgs::new(vec![low, high], len),
            &mut ctx,
        )
    }

    fn assert_any_in_range(
        values: &[Scalar],
        low: ArrayRef,
        high: ArrayRef,
        expected: &[bool],
    ) -> VortexResult<()> {
        let actual = execute(values, low, high)?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter(expected.iter().copied()),
            &mut ctx
        );
        Ok(())
    }

    /// Even values `0, 2, ..., 98`: a list of 50 whose bounds span every zone below.
    fn evens() -> Vec<Scalar> {
        (0..50).rev().map(|value| Scalar::from(value * 2)).collect()
    }

    #[rstest]
    // A zone strictly between two consecutive values is pruned.
    #[case(41, 41, false)]
    // A zone holding exactly one value is kept.
    #[case(41, 43, true)]
    // Bounds equal to a value are inclusive.
    #[case(42, 42, true)]
    #[case(40, 41, true)]
    #[case(39, 40, true)]
    // Zones beyond either end of the list.
    #[case(-5, -1, false)]
    #[case(-5, 0, true)]
    #[case(98, 200, true)]
    #[case(99, 200, false)]
    fn primitive_ranges(
        #[case] low: i32,
        #[case] high: i32,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        assert_any_in_range(
            &evens(),
            PrimitiveArray::from_iter([low]).into_array(),
            PrimitiveArray::from_iter([high]).into_array(),
            &[expected],
        )
    }

    #[test]
    fn null_bounds_may_match() -> VortexResult<()> {
        assert_any_in_range(
            &evens(),
            PrimitiveArray::from_option_iter([None, Some(41), Some(41), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(41), None, Some(41), None]).into_array(),
            &[true, true, false, true],
        )
    }

    #[test]
    fn options_drop_nulls_and_nans_and_sort() {
        let values = [
            Scalar::from(Some(3.0f64)),
            Scalar::from(Some(f64::NAN)),
            Scalar::null(DType::Primitive(
                vortex_array::dtype::PType::F64,
                Nullability::Nullable,
            )),
            Scalar::from(Some(1.0f64)),
            Scalar::from(Some(3.0f64)),
        ];
        let options = AnyInRangeOptions::try_new(&values).expect("values should be ordered");
        assert_eq!(
            options.values(),
            [Scalar::from(Some(1.0f64)), Scalar::from(Some(3.0f64))]
        );

        let only_nulls = [Scalar::null(DType::Primitive(
            vortex_array::dtype::PType::I32,
            Nullability::Nullable,
        ))];
        assert!(AnyInRangeOptions::try_new(&only_nulls).is_none());
    }

    #[test]
    fn float_nan_bounds_may_match() -> VortexResult<()> {
        let values: Vec<Scalar> = (0..20)
            .map(|value| Scalar::from(f64::from(value)))
            .collect();
        assert_any_in_range(
            &values,
            PrimitiveArray::from_iter([0.5f64, f64::NAN, 0.5, -0.0]).into_array(),
            PrimitiveArray::from_iter([0.75f64, 0.75, f64::NAN, -0.0]).into_array(),
            &[false, true, true, true],
        )
    }

    #[test]
    fn strings() -> VortexResult<()> {
        let values: Vec<Scalar> = ["kiwi", "apple", "fig", "cherry", "banana"]
            .into_iter()
            .map(Scalar::from)
            .collect();
        assert_any_in_range(
            &values,
            VarBinViewArray::from_iter_str(["apricot", "apple", "date", "fig", "a", "kiwis"])
                .into_array(),
            VarBinViewArray::from_iter_str(["bananas", "apple", "eggplant", "figs", "ap", "z"])
                .into_array(),
            &[true, true, false, true, false, false],
        )
    }

    #[test]
    fn strings_with_null_bounds() -> VortexResult<()> {
        let values: Vec<Scalar> = ["b", "d"].into_iter().map(Scalar::from).collect();
        assert_any_in_range(
            &values,
            VarBinViewArray::from_iter_nullable_str([None, Some("c")]).into_array(),
            VarBinViewArray::from_iter_nullable_str([Some("a"), Some("c")]).into_array(),
            &[true, false],
        )
    }

    #[test]
    fn decimals() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let values: Vec<Scalar> = [100i64, 300, 500]
            .into_iter()
            .map(|value| {
                Scalar::decimal(
                    DecimalValue::I64(value),
                    decimal_dtype,
                    Nullability::NonNullable,
                )
            })
            .collect();
        assert_any_in_range(
            &values,
            DecimalArray::from_option_iter([Some(101i64), Some(250), None], decimal_dtype)
                .into_array(),
            DecimalArray::from_option_iter([Some(299i64), Some(300), Some(0)], decimal_dtype)
                .into_array(),
            &[false, true, true],
        )
    }
}
