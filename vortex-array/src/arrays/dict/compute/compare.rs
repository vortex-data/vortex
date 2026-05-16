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
/// For sorted values, the predicate `value OP scalar` partitions the values into a contiguous
/// range. Translate the predicate into a single code-domain comparison and run it against the
/// codes child directly.
pub(crate) fn sorted_compare_scalar(
    lhs: ArrayView<'_, Dict>,
    scalar: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let values = lhs.values();
    let codes = lhs.codes().clone();
    let codes_len = codes.len();

    let result_nullability = codes.dtype().nullability();

    // Helper: emit a constant bool of length `codes_len` with the codes' nullability.
    let const_bool = |b: bool| -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(Scalar::bool(b, result_nullability), codes_len).into_array())
    };

    let values_ref: ArrayRef = values.clone();
    let left = values_ref.search_sorted(scalar, SearchSortedSide::Left)?;
    let right = values_ref.search_sorted(scalar, SearchSortedSide::Right)?;

    // Codes form an order-preserving encoding, so the predicate translates as follows.
    // Use `Left` for `<`-style boundaries (first index >= scalar) and `Right` for `>`-style
    // (first index > scalar).
    let (codes_op, codes_threshold) = match operator {
        CompareOperator::Eq => match left {
            SearchResult::Found(i) => (Operator::Eq, i),
            SearchResult::NotFound(_) => return const_bool(false).map(Some),
        },
        CompareOperator::NotEq => match left {
            SearchResult::Found(i) => (Operator::NotEq, i),
            SearchResult::NotFound(_) => {
                // All values != scalar; codes are also != scalar.
                // But preserve nulls: result is null where the code is null.
                if matches!(result_nullability, Nullability::Nullable) {
                    // codes != codes.max + 1 will be true wherever codes is non-null.
                    // Simpler: just const bool true preserving validity isn't trivial without
                    // a per-code mask. Fall back to the slow path.
                    return Ok(None);
                }
                return const_bool(true).map(Some);
            }
        },
        // value < scalar  iff  code < left
        CompareOperator::Lt => (Operator::Lt, left.to_index()),
        // value <= scalar iff  code < right
        CompareOperator::Lte => (Operator::Lt, right.to_index()),
        // value > scalar  iff  code >= right
        CompareOperator::Gt => (Operator::Gte, right.to_index()),
        // value >= scalar iff  code >= left
        CompareOperator::Gte => (Operator::Gte, left.to_index()),
    };

    // Build a same-typed scalar matching the codes ptype.
    let codes_threshold_scalar = code_threshold_scalar(&codes, codes_threshold)?;
    let threshold_arr =
        ConstantArray::new(codes_threshold_scalar, codes_len).into_array();
    let result = codes.binary(threshold_arr, codes_op)?;
    Ok(Some(result.execute::<Canonical>(ctx)?.into_array()))
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
        // Vortex guarantees codes are unsigned; signed codes are external compat only.
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
