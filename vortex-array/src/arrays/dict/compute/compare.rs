// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;

impl CompareKernel for Dict {
    fn compare(
        lhs: ArrayView<'_, Dict>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Sorted-values fast path: convert the predicate against values into a code-domain
        // comparison via binary search on the values array. Only applies when:
        //   - the dict is tagged sorted_values
        //   - the rhs is a constant
        //   - the values array is non-nullable (so the null prefix doesn't poison the result)
        if let Some(rhs_const) = rhs.as_constant()
            && lhs.has_sorted_values()
            && !lhs.values().dtype().is_nullable()
            && !rhs_const.is_null()
            && let Some(result) = sorted_compare_scalar(lhs, &rhs_const, operator, ctx)?
        {
            return Ok(Some(result));
        }

        // Original path: if we have more values than codes, it is faster to canonicalize first.
        if lhs.values().len() > lhs.codes().len() {
            return Ok(None);
        }

        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(rhs) = rhs.as_constant() {
            let compare_result = lhs.values().clone().binary(
                ConstantArray::new(rhs, lhs.values().len()).into_array(),
                Operator::from(operator),
            )?;

            // SAFETY: values len preserved, codes all still point to valid values
            let result = unsafe {
                DictArray::new_unchecked(lhs.codes().clone(), compare_result)
                    .set_all_values_referenced(lhs.has_all_values_referenced())
                    .into_array()
            };

            // We canonicalize the result because dictionary-encoded bools is dumb.
            return Ok(Some(result.execute::<Canonical>(ctx)?.into_array()));
        }

        Ok(None)
    }
}

/// Sorted-values fast path for comparing a dict to a constant.
///
/// For sorted values, the predicate `value OP scalar` partitions the values into a
/// contiguous range. We resolve that range via one or two binary searches on the values
/// array, build a tiny `BoolArray` of length `dict_len` representing the predicate per
/// dictionary slot, then dict-wrap and canonicalize. This is the same pipeline as the
/// existing path (`take(bool, codes)`), except the values-side comparison is replaced by
/// O(log dict_len) instead of O(dict_len).
pub(crate) fn sorted_compare_scalar(
    lhs: ArrayView<'_, Dict>,
    scalar: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let values = lhs.values().clone();
    let codes = lhs.codes().clone();
    let codes_len = codes.len();
    let dict_len = values.len();
    let result_nullability = codes.dtype().nullability();

    let const_bool = |b: bool| -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(Scalar::bool(b, result_nullability), codes_len).into_array())
    };

    let left = values.search_sorted(scalar, SearchSortedSide::Left)?;
    let right = values.search_sorted(scalar, SearchSortedSide::Right)?;

    // Resolve the predicate to a half-open range `[lo, hi)` of dictionary slots
    // for which the predicate is `true`.
    let (lo, hi): (usize, usize) = match operator {
        CompareOperator::Eq => match left {
            SearchResult::Found(i) => (i, i + 1),
            SearchResult::NotFound(_) => return const_bool(false).map(Some),
        },
        CompareOperator::NotEq => match left {
            SearchResult::Found(i) => {
                // True everywhere except slot `i`. Two ranges: [0,i) ∪ [i+1, dict_len).
                // Build the boolean directly below.
                let bool_values = build_two_range_bool(dict_len, i)?;
                return wrap_and_canonicalize(codes, bool_values, ctx).map(Some);
            }
            SearchResult::NotFound(_) => {
                if matches!(result_nullability, Nullability::Nullable) {
                    // Nulls in codes must be preserved; fall back to the existing path
                    // which handles that via DictArray<codes, bool_values>.
                    return Ok(None);
                }
                return const_bool(true).map(Some);
            }
        },
        CompareOperator::Lt => (0, left.to_index()),
        CompareOperator::Lte => (0, right.to_index()),
        CompareOperator::Gt => (right.to_index(), dict_len),
        CompareOperator::Gte => (left.to_index(), dict_len),
    };

    if lo >= hi {
        return const_bool(false).map(Some);
    }
    if lo == 0 && hi == dict_len {
        // All-true. Same caveat as NotEq above for nullable codes; the take/dict-wrap path
        // preserves nulls correctly when we go through it.
        if !matches!(result_nullability, Nullability::Nullable) {
            return const_bool(true).map(Some);
        }
    }

    let bool_values = build_range_bool(dict_len, lo, hi)?;
    wrap_and_canonicalize(codes, bool_values, ctx).map(Some)
}

/// Build a non-nullable `BoolArray` of `dict_len` where the range `[lo, hi)` is true.
fn build_range_bool(dict_len: usize, lo: usize, hi: usize) -> VortexResult<ArrayRef> {
    use vortex_buffer::BitBufferMut;
    let mut bb = BitBufferMut::with_capacity(dict_len);
    // Append `lo` false, `hi-lo` true, `dict_len-hi` false.
    bb.append_n(false, lo);
    bb.append_n(true, hi - lo);
    bb.append_n(false, dict_len - hi);
    use crate::arrays::BoolArray;
    use crate::validity::Validity;
    Ok(BoolArray::new(bb.freeze(), Validity::NonNullable).into_array())
}

/// Build a non-nullable `BoolArray` of `dict_len` where every slot is true except `skip_idx`.
fn build_two_range_bool(dict_len: usize, skip_idx: usize) -> VortexResult<ArrayRef> {
    use vortex_buffer::BitBufferMut;
    let mut bb = BitBufferMut::with_capacity(dict_len);
    bb.append_n(true, skip_idx);
    bb.append_false();
    bb.append_n(true, dict_len - skip_idx - 1);
    use crate::arrays::BoolArray;
    use crate::validity::Validity;
    Ok(BoolArray::new(bb.freeze(), Validity::NonNullable).into_array())
}

/// Wrap a small per-dict bool array as a `DictArray<codes, bool_values>` and canonicalize.
/// This reuses the highly-optimized take(BoolArray, codes) path.
fn wrap_and_canonicalize(
    codes: ArrayRef,
    bool_values: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // SAFETY: bool_values len equals dict_len, codes already index into it correctly.
    let dict = unsafe { DictArray::new_unchecked(codes, bool_values).into_array() };
    Ok(dict.execute::<Canonical>(ctx)?.into_array())
}

/// Build a `Scalar` of the codes' ptype holding `idx`. `idx` is guaranteed to be in
/// `0..=dict_len` and `dict_len` already fits in the codes ptype by DictArray invariant.
pub(crate) fn code_threshold_scalar(codes: &ArrayRef, idx: usize) -> VortexResult<Scalar> {
    use crate::dtype::DType;
    use crate::dtype::PType;
    let nullability = codes.dtype().nullability();
    match codes.dtype() {
        DType::Primitive(PType::U8, _) => Ok(Scalar::primitive(idx as u8, nullability)),
        DType::Primitive(PType::U16, _) => Ok(Scalar::primitive(idx as u16, nullability)),
        DType::Primitive(PType::U32, _) => Ok(Scalar::primitive(idx as u32, nullability)),
        DType::Primitive(PType::U64, _) => Ok(Scalar::primitive(idx as u64, nullability)),
        DType::Primitive(PType::I8, _) => Ok(Scalar::primitive(idx as i8, nullability)),
        DType::Primitive(PType::I16, _) => Ok(Scalar::primitive(idx as i16, nullability)),
        DType::Primitive(PType::I32, _) => Ok(Scalar::primitive(idx as i32, nullability)),
        DType::Primitive(PType::I64, _) => Ok(Scalar::primitive(idx as i64, nullability)),
        other => vortex_error::vortex_bail!("dict codes have unexpected dtype {other}"),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::builders::dict::dict_encode_sorted;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar_fn::fns::operators::Operator;

    fn cmp(dict: ArrayRef, scalar: ArrayRef, op: Operator) -> VortexResult<ArrayRef> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        dict.binary(scalar, op)?.execute::<Canonical>(&mut ctx).map(Into::into)
    }

    #[test]
    fn sorted_dict_eq_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Eq)?;
        assert_arrays_eq!(r, BoolArray::from_iter([false, false, true, false, false, true]));
        Ok(())
    }

    #[test]
    fn sorted_dict_eq_no_match() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(42i32, 3).into_array();
        let r = cmp(dict, scalar, Operator::Eq)?;
        assert_arrays_eq!(r, BoolArray::from_iter([false, false, false]));
        Ok(())
    }

    #[test]
    fn sorted_dict_lt_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(3i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Lt)?;
        assert_arrays_eq!(r, BoolArray::from_iter([false, true, true, true, false, true]));
        Ok(())
    }

    #[test]
    fn sorted_dict_gte_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Gte)?;
        assert_arrays_eq!(r, BoolArray::from_iter([true, false, true, false, true, true]));
        Ok(())
    }

    #[test]
    fn sorted_dict_lte_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Lte)?;
        assert_arrays_eq!(r, BoolArray::from_iter([false, true, true, true, false, true]));
        Ok(())
    }

    #[test]
    fn sorted_dict_gt_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Gt)?;
        assert_arrays_eq!(r, BoolArray::from_iter([true, false, false, false, true, false]));
        Ok(())
    }

    #[test]
    fn sorted_dict_eq_string() -> VortexResult<()> {
        let arr = VarBinArray::from_iter(
            [Some("zeta"), Some("alpha"), Some("mu"), Some("alpha"), Some("zeta")],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new("alpha", 5).into_array();
        let r = cmp(dict, scalar, Operator::Eq)?;
        assert_arrays_eq!(r, BoolArray::from_iter([false, true, false, true, false]));
        Ok(())
    }

    #[test]
    fn sorted_dict_gt_string() -> VortexResult<()> {
        let arr = VarBinArray::from_iter(
            [Some("zeta"), Some("alpha"), Some("mu"), Some("alpha"), Some("zeta")],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new("mu", 5).into_array();
        let r = cmp(dict, scalar, Operator::Gt)?;
        assert_arrays_eq!(r, BoolArray::from_iter([true, false, false, false, true]));
        Ok(())
    }
}
