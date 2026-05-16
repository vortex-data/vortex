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
/// contiguous code range. Translate the predicate into a single (or two-clause) integer
/// comparison in the code domain and run it directly against the codes child via Arrow's
/// vectorized primitive compare kernel. This skips the `take(bool_values, codes)` step
/// that the plain path needs — codes go straight to the result.
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

    // Typed binary search avoids the generic `IndexOrd<Scalar>` path's per-probe
    // execute_scalar overhead.
    let (left, right) = sorted_compare_scalar_search(&values, scalar)?;

    // Translate the predicate to a code-domain operator + threshold (or pair).
    use crate::scalar_fn::fns::operators::Operator;
    match operator {
        CompareOperator::Eq => match left {
            SearchResult::Found(i) => {
                // codes == i (preserves codes' nullability automatically).
                code_cmp(&codes, i, Operator::Eq, ctx).map(Some)
            }
            SearchResult::NotFound(_) => const_bool(false).map(Some),
        },
        CompareOperator::NotEq => match left {
            SearchResult::Found(i) => code_cmp(&codes, i, Operator::NotEq, ctx).map(Some),
            SearchResult::NotFound(_) => {
                // All values != scalar. For nullable codes the result must be null where
                // codes are null; fall back to the existing path which handles that.
                if matches!(result_nullability, Nullability::Nullable) {
                    Ok(None)
                } else {
                    const_bool(true).map(Some)
                }
            }
        },
        // value < scalar  iff  code < left  iff  code <= left-1   (left can be 0 → all false)
        CompareOperator::Lt => {
            let bound = left.to_index();
            if bound == 0 {
                return const_bool(false).map(Some);
            }
            if bound >= dict_len {
                if !matches!(result_nullability, Nullability::Nullable) {
                    return const_bool(true).map(Some);
                }
            }
            code_cmp(&codes, bound, Operator::Lt, ctx).map(Some)
        }
        // value <= scalar iff  code < right
        CompareOperator::Lte => {
            let bound = right.to_index();
            if bound == 0 {
                return const_bool(false).map(Some);
            }
            if bound >= dict_len {
                if !matches!(result_nullability, Nullability::Nullable) {
                    return const_bool(true).map(Some);
                }
            }
            code_cmp(&codes, bound, Operator::Lt, ctx).map(Some)
        }
        // value > scalar  iff  code >= right
        CompareOperator::Gt => {
            let bound = right.to_index();
            if bound == 0 {
                if !matches!(result_nullability, Nullability::Nullable) {
                    return const_bool(true).map(Some);
                }
            }
            if bound >= dict_len {
                return const_bool(false).map(Some);
            }
            code_cmp(&codes, bound, Operator::Gte, ctx).map(Some)
        }
        // value >= scalar iff  code >= left
        CompareOperator::Gte => {
            let bound = left.to_index();
            if bound == 0 {
                if !matches!(result_nullability, Nullability::Nullable) {
                    return const_bool(true).map(Some);
                }
            }
            if bound >= dict_len {
                return const_bool(false).map(Some);
            }
            code_cmp(&codes, bound, Operator::Gte, ctx).map(Some)
        }
    }
}

/// Run a primitive code-domain compare against an integer threshold. Wraps the threshold
/// in a `ConstantArray` (O(1) construction) so Arrow's vectorized kernel can scalar-broadcast.
pub(crate) fn code_cmp(
    codes: &ArrayRef,
    threshold: usize,
    op: Operator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let threshold_scalar = code_threshold_scalar(codes, threshold)?;
    let len = codes.len();
    let threshold_arr = ConstantArray::new(threshold_scalar, len).into_array();
    codes
        .clone()
        .binary(threshold_arr, op)?
        .execute::<Canonical>(ctx)
        .map(Into::into)
}

/// Build a `Scalar` of the codes' ptype holding `idx`. `idx` is in `0..=dict_len` and
/// `dict_len` already fits in the codes ptype by DictArray invariant.
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

/// Resolve `(left, right)` search-sorted boundaries for `scalar` against `values`,
/// preferring the typed fast path and falling back to the generic `IndexOrd<Scalar>` path.
pub(crate) fn sorted_compare_scalar_search(
    values: &ArrayRef,
    scalar: &Scalar,
) -> VortexResult<(SearchResult, SearchResult)> {
    if let Some(pair) = typed_search_pair(values, scalar)? {
        return Ok(pair);
    }
    Ok((
        values.search_sorted(scalar, SearchSortedSide::Left)?,
        values.search_sorted(scalar, SearchSortedSide::Right)?,
    ))
}

/// Typed binary search on the values array. Returns `(left, right)` boundaries if `values`
/// is a canonical Primitive or VarBinView and a needle of the matching native type can be
/// extracted from `scalar`. Returns `None` if the typed fast path doesn't apply, in which
/// case the caller should fall back to the generic `IndexOrd<Scalar>` path.
fn typed_search_pair(
    values: &ArrayRef,
    scalar: &Scalar,
) -> VortexResult<Option<(SearchResult, SearchResult)>> {
    use crate::accessor::ArrayAccessor;
    use crate::arrays::Primitive;
    use crate::arrays::VarBinView;
    use crate::match_each_native_ptype;

    if let Some(prim) = values.as_opt::<Primitive>() {
        return Ok(Some(match_each_native_ptype!(prim.ptype(), |T| {
            let Ok(needle) = T::try_from(scalar) else {
                return Ok(None);
            };
            typed_search_primitive::<T>(prim.as_slice::<T>(), needle)
        })));
    }
    if let Some(vbv) = values.as_opt::<VarBinView>() {
        let needle_bytes: Vec<u8> = match scalar.dtype() {
            crate::dtype::DType::Utf8(_) => {
                let s: Option<String> = scalar.try_into().ok();
                let Some(s) = s else { return Ok(None) };
                s.into_bytes()
            }
            crate::dtype::DType::Binary(_) => {
                let b: Option<Vec<u8>> = scalar.try_into().ok();
                let Some(b) = b else { return Ok(None) };
                b
            }
            _ => return Ok(None),
        };
        // Materialize the dict's values once as Vec<Option<&[u8]>> — dict size is bounded
        // by u16::MAX so this is cheap and avoids repeated view resolution per probe.
        let resolved: Vec<Option<Vec<u8>>> = vbv
            .into_owned()
            .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
                it.map(|opt| opt.map(|b| b.to_vec())).collect()
            });
        return Ok(Some(typed_search_bytes(&resolved, &needle_bytes)));
    }
    Ok(None)
}

fn typed_search_primitive<T: crate::dtype::NativePType>(
    slice: &[T],
    needle: T,
) -> (SearchResult, SearchResult) {
    use std::cmp::Ordering::*;
    let left = match slice.binary_search_by(|probe| match probe.total_compare(needle) {
        Equal => Greater,
        other => other,
    }) {
        Ok(_) => unreachable!("custom comparator never returns Equal"),
        Err(i) => {
            // i is the first index where probe >= needle.
            if i < slice.len() && slice[i].total_compare(needle) == Equal {
                SearchResult::Found(i)
            } else {
                SearchResult::NotFound(i)
            }
        }
    };
    let right = match slice.binary_search_by(|probe| match probe.total_compare(needle) {
        Equal => Less,
        other => other,
    }) {
        Ok(_) => unreachable!("custom comparator never returns Equal"),
        Err(i) => {
            // i is the first index where probe > needle.
            if i > 0 && slice[i - 1].total_compare(needle) == Equal {
                SearchResult::Found(i)
            } else {
                SearchResult::NotFound(i)
            }
        }
    };
    (left, right)
}

fn typed_search_bytes(
    resolved: &[Option<Vec<u8>>],
    needle: &[u8],
) -> (SearchResult, SearchResult) {
    // Convention: nulls sort first, so for any non-null needle, the null prefix is less.
    let cmp = |probe: &Option<Vec<u8>>| -> std::cmp::Ordering {
        match probe {
            None => std::cmp::Ordering::Less,
            Some(v) => v.as_slice().cmp(needle),
        }
    };
    let left = match resolved.binary_search_by(|probe| match cmp(probe) {
        std::cmp::Ordering::Equal => std::cmp::Ordering::Greater,
        other => other,
    }) {
        Ok(_) => unreachable!("custom comparator never returns Equal"),
        Err(i) => {
            if i < resolved.len() && cmp(&resolved[i]) == std::cmp::Ordering::Equal {
                SearchResult::Found(i)
            } else {
                SearchResult::NotFound(i)
            }
        }
    };
    let right = match resolved.binary_search_by(|probe| match cmp(probe) {
        std::cmp::Ordering::Equal => std::cmp::Ordering::Less,
        other => other,
    }) {
        Ok(_) => unreachable!("custom comparator never returns Equal"),
        Err(i) => {
            if i > 0 && cmp(&resolved[i - 1]) == std::cmp::Ordering::Equal {
                SearchResult::Found(i)
            } else {
                SearchResult::NotFound(i)
            }
        }
    };
    (left, right)
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
