// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparison of a sorted array against a constant by binary search.
//!
//! An array with an exact cached `IsSorted` or `IsStrictSorted` statistic holds its nulls as a
//! prefix followed by its valid values in non-decreasing order. Comparing such an array against a
//! constant selects at most two contiguous ranges, so two binary searches (one for a strictly
//! sorted array) replace the linear scan.
//!
//! Only integers (including integer-backed extensions such as timestamps), UTF-8 and binary are
//! handled: their sort order agrees with the comparison kernels without float total-ordering
//! caveats.

use std::cmp::Ordering;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::VarBinView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::bytes::constant_bytes;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::binary::compare::extension_storage;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

/// Compare a sorted array against a constant, in either operand order.
///
/// Returns `None` when neither operand is a non-null constant, the other operand has no exact
/// cached sortedness statistic, or its dtype is not supported.
pub(crate) fn compare_sorted_constant(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let (array, constant, op) = if let Some(constant) = rhs.as_opt::<Constant>() {
        (lhs, constant.scalar().clone(), op)
    } else if let Some(constant) = lhs.as_opt::<Constant>() {
        (rhs, constant.scalar().clone(), op.swap())
    } else {
        return Ok(None);
    };

    if array.is::<Constant>()
        || constant.is_null()
        || lhs.len() != rhs.len()
        || !array.dtype().eq_ignore_nullability(constant.dtype())
    {
        return Ok(None);
    }

    let Some(strict) = cached_sortedness(array) else {
        return Ok(None);
    };

    let nullability =
        Nullability::from(array.dtype().is_nullable() || constant.dtype().is_nullable());
    let constant_validity = if constant.dtype().is_nullable() {
        Validity::AllValid
    } else {
        Validity::NonNullable
    };

    // Extensions with an integer storage type (timestamps, dates, times) sort by their storage.
    let (array, constant) = if array.dtype().is_extension() {
        (
            extension_storage(array, ctx)?,
            constant.as_extension().to_storage_scalar(),
        )
    } else {
        (array.clone(), constant)
    };

    match array.dtype() {
        DType::Primitive(ptype, _) if ptype.is_int() => {}
        DType::Utf8(_) | DType::Binary(_) => {}
        _ => return Ok(None),
    }
    let len = array.len();
    let bounds = sorted_bounds(&array, &constant, strict, ctx)?;

    if let Some(value) = bounds.constant(op, len) {
        let scalar = match value {
            Some(value) => Scalar::bool(value, nullability),
            None => Scalar::null(DType::Bool(nullability)),
        };
        return Ok(Some(ConstantArray::new(scalar, len).into_array()));
    }

    let bits = bounds.select(op, len, ctx.allocator());
    let validity = compare_validity(array.validity()?, constant_validity, nullability)?;
    Ok(Some(BoolArray::try_new(bits, validity)?.into_array()))
}

/// The exact cached sortedness of `array`: `Some(true)` if strictly sorted, `Some(false)` if
/// sorted, and `None` if unknown or unsorted.
///
/// The statistic is never computed here: doing so costs a full scan, which is what the binary
/// search avoids.
fn cached_sortedness(array: &ArrayRef) -> Option<bool> {
    let stats = array.statistics();
    if stats.get_as::<bool>(Stat::IsStrictSorted) == Precision::Exact(true) {
        return Some(true);
    }
    if stats.get_as::<bool>(Stat::IsSorted) == Precision::Exact(true) {
        return Some(false);
    }
    None
}

fn sorted_bounds(
    array: &ArrayRef,
    constant: &Scalar,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<SortedBounds> {
    let nulls = array.invalid_count(ctx)?;
    if let Some(primitive) = array.as_opt::<Primitive>() {
        return match_each_integer_ptype!(primitive.ptype(), |T| {
            let value = constant
                .as_primitive()
                .try_typed_value::<T>()?
                .ok_or_else(|| vortex_err!("null constant handled by compare_sorted_constant"))?;
            let values = primitive.as_slice::<T>();
            SortedBounds::search(nulls, values.len(), strict, |idx| {
                Ok(values[idx].cmp(&value))
            })
        });
    }
    if let Some(views) = array.as_opt::<VarBinView>() {
        let value = constant_bytes(constant)?;
        let views = views.into_owned();
        let side = views.resolved_views();
        return SortedBounds::search(nulls, side.len(), strict, |idx| {
            Ok(side.bytes(idx).cmp(value.as_slice()))
        });
    }
    // Probe encoded arrays in O(log n) scalar accesses without decoding the whole array.
    SortedBounds::search(nulls, array.len(), strict, |idx| {
        array
            .execute_scalar(idx, ctx)?
            .partial_cmp(constant)
            .ok_or_else(|| {
                vortex_err!("Cannot compare {} with {}", array.dtype(), constant.dtype())
            })
    })
}

/// The range of a sorted array whose valid values equal the searched constant.
///
/// Valid values occupy `nulls..len`; values in `nulls..lower` are less than the constant,
/// `lower..upper` are equal, and `upper..len` are greater.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SortedBounds {
    nulls: usize,
    lower: usize,
    upper: usize,
}

impl SortedBounds {
    /// Locate the constant in `nulls..len`, where `cmp(idx)` orders the value at `idx` against
    /// the constant.
    fn search(
        nulls: usize,
        len: usize,
        strict: bool,
        mut cmp: impl FnMut(usize) -> VortexResult<Ordering>,
    ) -> VortexResult<Self> {
        let lower = partition_point(nulls, len, |idx| Ok(cmp(idx)?.is_lt()))?;
        // A strictly sorted array holds at most one value equal to the constant.
        let upper = if strict {
            lower + usize::from(lower < len && cmp(lower)?.is_eq())
        } else {
            partition_point(lower, len, |idx| Ok(cmp(idx)?.is_le()))?
        };
        Ok(Self {
            nulls,
            lower,
            upper,
        })
    }

    /// The `(unset, set)` run-length pairs of the result bits, in ascending order. Null positions
    /// are left unset; the result validity masks them.
    fn runs(self, op: CompareOperator, len: usize) -> [(usize, usize); 2] {
        let Self {
            nulls,
            lower,
            upper,
        } = self;
        match op {
            CompareOperator::Lt => [(nulls, lower - nulls), (0, 0)],
            CompareOperator::Lte => [(nulls, upper - nulls), (0, 0)],
            CompareOperator::Gt => [(upper, len - upper), (0, 0)],
            CompareOperator::Gte => [(lower, len - lower), (0, 0)],
            CompareOperator::Eq => [(lower, upper - lower), (0, 0)],
            CompareOperator::NotEq => [(nulls, lower - nulls), (upper - lower, len - upper)],
        }
    }

    /// The result value when it is the same at every position: `Some(None)` if every value is
    /// null, `Some(Some(b))` if every value is `b`, and `None` if the result varies.
    fn constant(self, op: CompareOperator, len: usize) -> Option<Option<bool>> {
        if self.nulls == len {
            return Some(None);
        }
        if self.nulls > 0 {
            return None;
        }
        let set: usize = self.runs(op, len).iter().map(|(_, set)| set).sum();
        match set {
            0 => Some(Some(false)),
            set if set == len => Some(Some(true)),
            _ => None,
        }
    }

    /// The comparison result bits.
    fn select(self, op: CompareOperator, len: usize, allocator: &BufferAllocatorRef) -> BitBuffer {
        let runs = self.runs(op, len);

        let mut bits = BitBufferMut::with_capacity_in(len, allocator.clone());
        for (unset, set) in runs {
            bits.append_n(false, unset);
            bits.append_n(true, set);
        }
        bits.append_n(false, len - bits.len());
        bits.freeze()
    }
}

/// The first index in `lo..hi` for which `pred` is false, assuming `pred` is true for a prefix
/// of the range and false for the rest.
fn partition_point(
    mut lo: usize,
    mut hi: usize,
    mut pred: impl FnMut(usize) -> VortexResult<bool>,
) -> VortexResult<usize> {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid)? {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::scalar_fn::fns::binary::compare::scalar_cmp;
    use crate::scalar_fn::fns::binary::compare::sorted::compare_sorted_constant;
    use crate::scalar_fn::fns::operators::CompareOperator;

    const OPS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    fn mark_sorted(array: &ArrayRef, strict: bool) {
        let stat = if strict {
            Stat::IsStrictSorted
        } else {
            Stat::IsSorted
        };
        array
            .statistics()
            .set(stat, Precision::exact(ScalarValue::from(true)));
    }

    /// Row-wise reference result: compare each scalar against the constant.
    fn expected(
        array: &ArrayRef,
        constant: &Scalar,
        op: CompareOperator,
    ) -> VortexResult<BoolArray> {
        let mut ctx = array_session().create_execution_ctx();
        let values = (0..array.len())
            .map(|idx| {
                let value = array.execute_scalar(idx, &mut ctx)?;
                Ok(scalar_cmp(&value, constant, op)?.as_bool().value())
            })
            .collect::<VortexResult<Vec<Option<bool>>>>()?;
        Ok(if array.dtype().is_nullable() {
            BoolArray::from_iter(values)
        } else {
            BoolArray::from_iter(values.into_iter().map(|v| v.unwrap_or(false)))
        })
    }

    /// Check every operator against the row-wise reference in both operand orders.
    fn check_all_ops(array: &ArrayRef, constant: Scalar, strict: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        mark_sorted(array, strict);
        let constant_array = ConstantArray::new(constant.clone(), array.len()).into_array();
        for op in OPS {
            let expected = expected(array, &constant, op)?;

            let result = compare_sorted_constant(array, &constant_array, op, &mut ctx)?
                .expect("sorted fast path applies");
            assert_arrays_eq!(result, expected.clone(), &mut ctx);

            let swapped = compare_sorted_constant(&constant_array, array, op.swap(), &mut ctx)?
                .expect("sorted fast path applies");
            assert_arrays_eq!(swapped, expected, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    #[case::below(-5)]
    #[case::first(1)]
    #[case::duplicate(3)]
    #[case::gap(4)]
    #[case::last(9)]
    #[case::above(100)]
    fn sorted_ints(#[case] needle: i32) -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 3, 3, 5, 8, 9, 9].into_array();
        check_all_ops(&array, Scalar::from(needle), false)
    }

    #[rstest]
    #[case::below(0)]
    #[case::first(1)]
    #[case::present(5)]
    #[case::gap(4)]
    #[case::last(9)]
    #[case::above(100)]
    fn strict_sorted_ints(#[case] needle: u64) -> VortexResult<()> {
        let array = buffer![1u64, 2, 3, 5, 8, 9].into_array();
        check_all_ops(&array, Scalar::from(needle), true)
    }

    #[rstest]
    #[case::sorted(false)]
    #[case::strict(true)]
    fn nullable_ints_null_prefix(#[case] strict: bool) -> VortexResult<()> {
        let values = if strict {
            vec![None, Some(1i64), Some(4), Some(7)]
        } else {
            vec![None, None, Some(1i64), Some(4), Some(4), Some(7)]
        };
        let array = PrimitiveArray::from_option_iter(values).into_array();
        for needle in [0i64, 1, 4, 5, 7, 8] {
            check_all_ops(&array, Scalar::from(needle), strict)?;
        }
        Ok(())
    }

    #[rstest]
    #[case::sorted(false)]
    #[case::strict(true)]
    fn timestamps(#[case] strict: bool) -> VortexResult<()> {
        let storage = if strict {
            buffer![1_000i64, 2_000, 3_000, 4_000]
        } else {
            buffer![1_000i64, 2_000, 2_000, 4_000]
        };
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let array = ExtensionArray::new(ext_dtype.clone(), storage.into_array()).into_array();
        for needle in [0i64, 1_000, 2_000, 2_500, 4_000, 5_000] {
            let constant = Scalar::extension_ref(ext_dtype.clone(), Scalar::from(needle));
            check_all_ops(&array, constant, strict)?;
        }
        Ok(())
    }

    #[rstest]
    #[case::sorted(false)]
    #[case::strict(true)]
    fn strings(#[case] strict: bool) -> VortexResult<()> {
        let values: Vec<Option<&str>> = if strict {
            vec![
                None,
                Some("a"),
                Some("apple"),
                Some("banana"),
                Some("this is a long string"),
            ]
        } else {
            vec![
                None,
                None,
                Some("a"),
                Some("apple"),
                Some("apple"),
                Some("banana"),
                Some("this is a long string"),
                Some("this is a long string"),
            ]
        };
        let array = VarBinViewArray::from_iter_nullable_str(values).into_array();
        for needle in [
            "",
            "a",
            "ab",
            "apple",
            "banana",
            "this is a long string",
            "zzz",
        ] {
            check_all_ops(&array, Scalar::from(needle), strict)?;
        }
        Ok(())
    }

    /// A result that is the same at every position is returned as a constant, not a bitmap.
    #[rstest]
    #[case::all_true(CompareOperator::Lt, 100, Some(true))]
    #[case::all_false(CompareOperator::Gt, 100, Some(false))]
    #[case::none_equal(CompareOperator::Eq, 4, Some(false))]
    #[case::all_not_equal(CompareOperator::NotEq, -1, Some(true))]
    #[case::mixed(CompareOperator::Lt, 3, None)]
    fn uniform_result_is_constant(
        #[case] op: CompareOperator,
        #[case] needle: i32,
        #[case] expected: Option<bool>,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![1i32, 2, 3, 5].into_array();
        mark_sorted(&array, false);
        let constant = ConstantArray::new(needle, array.len()).into_array();
        let result = compare_sorted_constant(&array, &constant, op, &mut ctx)?
            .expect("sorted fast path applies");
        assert_eq!(
            result.as_constant().and_then(|s| s.as_bool().value()),
            expected
        );
        Ok(())
    }

    #[test]
    fn all_null_is_null_constant() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([None::<i32>, None]).into_array();
        mark_sorted(&array, false);
        let constant = ConstantArray::new(1i32, array.len()).into_array();
        let result = compare_sorted_constant(&array, &constant, CompareOperator::Lt, &mut ctx)?
            .expect("sorted fast path applies");
        assert!(result.as_constant().is_some_and(|s| s.is_null()));
        Ok(())
    }

    /// Arrays without a cached sortedness statistic fall through to the linear kernels.
    #[test]
    fn unsorted_falls_through() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![3i32, 1, 2].into_array();
        let constant = ConstantArray::new(2i32, 3).into_array();
        assert!(
            compare_sorted_constant(&array, &constant, CompareOperator::Lt, &mut ctx)?.is_none()
        );
        Ok(())
    }
}
