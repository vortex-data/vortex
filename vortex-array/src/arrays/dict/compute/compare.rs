// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

impl CompareKernel for Dict {
    fn compare(
        lhs: ArrayView<'_, Dict>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if lhs.values().len() > lhs.codes().len() {
            return Ok(None);
        }

        if let Some(rhs) = rhs.as_constant() {
            let compare_result = lhs.values().clone().binary(
                ConstantArray::new(rhs, lhs.values().len()).into_array(),
                Operator::from(operator),
            )?;

            // SAFETY: values len preserved, codes still point to valid values.
            let result = unsafe {
                DictArray::new_unchecked(lhs.codes().clone(), compare_result)
                    .set_all_values_referenced(lhs.has_all_values_referenced())
                    .into_array()
            };

            return Ok(Some(result.execute::<Canonical>(ctx)?.into_array()));
        }

        Ok(None)
    }
}

/// Rewrite a sorted-dict predicate `values OP scalar` as a codes-domain compare.
///
/// Resolves `scalar`'s position in the sorted values via a linear scan, then emits one of:
/// - `Ok(Some(ScalarFnArray(Binary, [codes, const])))` — codes-domain compare
/// - `Ok(Some(ConstantArray))` — short-circuit when the predicate is constant; the
///   canonical Mask folds to `AllTrue`/`AllFalse` in O(1)
/// - `Ok(None)` if the values aren't a typed scan target (caller falls back to push-down)
///
/// TODO(sorted-column): per `benches/dict_sorted_pushdowns.rs::cmp_*`, the remaining
/// cost is the SIMD compare over every code; the only way to beat it is to skip per-row
/// evaluation, which needs the *codes* to be sorted too (fully sorted column), in which
/// case the predicate collapses to `Mask::from_slices(N, [(0, k)])`.
pub(crate) fn reduce_sorted_compare(
    lhs: ArrayView<'_, Dict>,
    scalar: &Scalar,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    let values = lhs.values().clone();
    let codes = lhs.codes().clone();
    let codes_len = codes.len();
    let dict_len = values.len();
    let result_nullability = codes.dtype().nullability();

    let Some(bounds) = scan_sorted_bounds(&values, scalar)? else {
        return Ok(None);
    };

    let const_bool = |b: bool| -> ArrayRef {
        ConstantArray::new(Scalar::bool(b, result_nullability), codes_len).into_array()
    };

    match operator {
        CompareOperator::Eq => match bounds.found {
            Some(i) => Ok(Some(emit_code_cmp(&codes, i, Operator::Eq)?)),
            None => Ok(Some(const_bool(false))),
        },
        CompareOperator::NotEq => match bounds.found {
            Some(i) => Ok(Some(emit_code_cmp(&codes, i, Operator::NotEq)?)),
            None if matches!(result_nullability, Nullability::Nullable) => Ok(None),
            None => Ok(Some(const_bool(true))),
        },
        // value < scalar  iff  code < left   (left = first idx with value >= scalar)
        CompareOperator::Lt => {
            emit_bounded_cmp(&codes, bounds.left, dict_len, Operator::Lt, result_nullability, codes_len)
        }
        // value <= scalar iff  code < right  (right = first idx with value > scalar)
        CompareOperator::Lte => {
            emit_bounded_cmp(&codes, bounds.right, dict_len, Operator::Lt, result_nullability, codes_len)
        }
        // value > scalar  iff  code >= right
        CompareOperator::Gt => {
            emit_bounded_cmp(&codes, bounds.right, dict_len, Operator::Gte, result_nullability, codes_len)
        }
        // value >= scalar iff  code >= left
        CompareOperator::Gte => {
            emit_bounded_cmp(&codes, bounds.left, dict_len, Operator::Gte, result_nullability, codes_len)
        }
    }
}

/// Emit `code OP bound`, collapsing to a `ConstantArray<Bool>` when `bound` is at an
/// extreme. `op` must be `Lt` or `Gte`.
fn emit_bounded_cmp(
    codes: &ArrayRef,
    bound: usize,
    dict_len: usize,
    op: Operator,
    nullability: Nullability,
    codes_len: usize,
) -> VortexResult<Option<ArrayRef>> {
    let const_bool = |b: bool| -> ArrayRef {
        ConstantArray::new(Scalar::bool(b, nullability), codes_len).into_array()
    };
    // For Lt: `code < 0` is false; `code < dict_len` is true.
    // For Gte: `code >= 0` is true; `code >= dict_len` is false.
    let (at_zero, at_top) = match op {
        Operator::Lt => (false, true),
        Operator::Gte => (true, false),
        _ => vortex_error::vortex_panic!("emit_bounded_cmp expects Lt or Gte, got {op:?}"),
    };
    // The "always-true" arm needs non-nullable codes so nulls propagate correctly.
    if bound == 0 && (!at_zero || !nullability.is_nullable()) {
        return Ok(Some(const_bool(at_zero)));
    }
    if bound >= dict_len && (!at_top || !nullability.is_nullable()) {
        return Ok(Some(const_bool(at_top)));
    }
    Ok(Some(emit_code_cmp(codes, bound, op)?))
}

/// Emit `codes OP threshold`.
pub(crate) fn emit_code_cmp(
    codes: &ArrayRef,
    threshold: usize,
    op: Operator,
) -> VortexResult<ArrayRef> {
    let threshold_arr =
        ConstantArray::new(code_threshold_scalar(codes, threshold)?, codes.len()).into_array();
    codes.clone().binary(threshold_arr, op)
}

/// Build a `Scalar` of the codes' ptype holding `idx`. By DictArray invariant `dict_len`
/// (and therefore any `idx <= dict_len`) fits in the codes ptype.
pub(crate) fn code_threshold_scalar(codes: &ArrayRef, idx: usize) -> VortexResult<Scalar> {
    use crate::dtype::DType;
    use crate::match_each_integer_ptype;
    let nullability = codes.dtype().nullability();
    let DType::Primitive(ptype, _) = codes.dtype() else {
        vortex_error::vortex_bail!("dict codes have unexpected dtype {}", codes.dtype());
    };
    Ok(match_each_integer_ptype!(ptype, |T| {
        #[allow(clippy::cast_possible_truncation)]
        Scalar::primitive(idx as T, nullability)
    }))
}

/// Boundary indices on a sorted (ascending, non-nullable) values array.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SortedBounds {
    /// First index `i` where `values[i] >= scalar`.
    pub left: usize,
    /// First index `i` where `values[i] > scalar`.
    pub right: usize,
    /// `Some(i)` iff `values[i] == scalar`; `None` if no exact match.
    pub found: Option<usize>,
}

/// Scan a sorted values array to find boundaries for `scalar`. Dict size is bounded
/// (typically <= u16::MAX) so the typed linear scan beats binary_search + `IndexOrd`
/// scaffolding here.
///
/// Returns `None` if `values` isn't a canonical Primitive/VarBinView or `scalar` can't
/// convert to the matching native type.
pub(crate) fn scan_sorted_bounds(
    values: &ArrayRef,
    scalar: &Scalar,
) -> VortexResult<Option<SortedBounds>> {
    scan_sorted_dual_bounds(values, scalar, scalar).map(|opt| opt.map(|(b, _)| b))
}

/// Same as [`scan_sorted_bounds`] but resolves `lower` and `upper` in a single pass over
/// the values. Once the scan passes `upper`, neither bound can move again.
pub(crate) fn scan_sorted_dual_bounds(
    values: &ArrayRef,
    lower: &Scalar,
    upper: &Scalar,
) -> VortexResult<Option<(SortedBounds, SortedBounds)>> {
    use crate::accessor::ArrayAccessor;
    use crate::arrays::Primitive;
    use crate::arrays::VarBinView;
    use crate::match_each_native_ptype;

    if let Some(prim) = values.as_opt::<Primitive>() {
        return Ok(Some(match_each_native_ptype!(prim.ptype(), |T| {
            let (Ok(lo), Ok(hi)) = (T::try_from(lower), T::try_from(upper)) else {
                return Ok(None);
            };
            scan_primitive_dual::<T>(prim.as_slice::<T>(), lo, hi)
        })));
    }
    if let Some(vbv) = values.as_opt::<VarBinView>() {
        let (Some(lo), Some(hi)) = (scalar_as_bytes(lower), scalar_as_bytes(upper)) else {
            return Ok(None);
        };
        let bounds = vbv
            .into_owned()
            .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
                scan_bytes_dual(it, lo, hi)
            });
        return Ok(Some(bounds));
    }
    Ok(None)
}

fn scalar_as_bytes(scalar: &Scalar) -> Option<&[u8]> {
    use crate::dtype::DType;
    match scalar.dtype() {
        DType::Utf8(_) => scalar.as_utf8_opt().and_then(|v| v.value()).map(|s| s.as_bytes()),
        DType::Binary(_) => scalar.as_binary_opt().and_then(|v| v.value()).map(|b| b.as_slice()),
        _ => None,
    }
}

fn scan_primitive_dual<T: crate::dtype::NativePType>(
    slice: &[T],
    lo: T,
    hi: T,
) -> (SortedBounds, SortedBounds) {
    let (lo_bounds, exit) = scan_primitive_from(slice, 0, lo);
    if lo.total_compare(hi) == std::cmp::Ordering::Equal {
        return (lo_bounds, lo_bounds);
    }
    let (hi_bounds, _) = scan_primitive_from(slice, exit, hi);
    (lo_bounds, hi_bounds)
}

fn scan_bytes_dual(
    it: &mut dyn Iterator<Item = Option<&[u8]>>,
    lo: &[u8],
    hi: &[u8],
) -> (SortedBounds, SortedBounds) {
    let items: Vec<Option<&[u8]>> = it.collect();
    let (lo_bounds, exit) = scan_bytes_from(&items, 0, lo);
    if lo == hi {
        return (lo_bounds, lo_bounds);
    }
    let (hi_bounds, _) = scan_bytes_from(&items, exit, hi);
    (lo_bounds, hi_bounds)
}

/// Walk `slice[start..]`, returning the `(left, right, found)` boundaries for `needle`
/// and the exit index (where the scan stopped — `slice.len()` if it ran to the end).
fn scan_primitive_from<T: crate::dtype::NativePType>(
    slice: &[T],
    start: usize,
    needle: T,
) -> (SortedBounds, usize) {
    use std::cmp::Ordering::*;
    let len = slice.len();
    let mut left: Option<usize> = None;
    let mut right: Option<usize> = None;
    let mut found: Option<usize> = None;
    let mut exit = len;
    let mut idx = start;
    while idx < len {
        // SAFETY: idx < len.
        let v = unsafe { *slice.get_unchecked(idx) };
        match v.total_compare(needle) {
            Less => {}
            Equal => {
                if left.is_none() {
                    left = Some(idx);
                    found = Some(idx);
                }
            }
            Greater => {
                if left.is_none() {
                    left = Some(idx);
                }
                right = Some(idx);
                exit = idx;
                break;
            }
        }
        idx += 1;
    }
    (
        SortedBounds {
            left: left.unwrap_or(len),
            right: right.unwrap_or(len),
            found,
        },
        exit,
    )
}

fn scan_bytes_from(items: &[Option<&[u8]>], start: usize, needle: &[u8]) -> (SortedBounds, usize) {
    use std::cmp::Ordering::*;
    let len = items.len();
    let mut left: Option<usize> = None;
    let mut right: Option<usize> = None;
    let mut found: Option<usize> = None;
    let mut exit = len;
    for (idx, opt) in items.iter().copied().enumerate().skip(start) {
        let cmp = match opt {
            None => Less, // nulls sort first
            Some(b) => b.cmp(needle),
        };
        match cmp {
            Less => {}
            Equal => {
                if left.is_none() {
                    left = Some(idx);
                    found = Some(idx);
                }
            }
            Greater => {
                if left.is_none() {
                    left = Some(idx);
                }
                right = Some(idx);
                exit = idx;
                break;
            }
        }
    }
    (
        SortedBounds {
            left: left.unwrap_or(len),
            right: right.unwrap_or(len),
            found,
        },
        exit,
    )
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
        dict.binary(scalar, op)?
            .execute::<Canonical>(&mut ctx)
            .map(Into::into)
    }

    #[test]
    fn sorted_dict_eq_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Eq)?;
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([false, false, true, false, false, true])
        );
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
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([false, true, true, true, false, true])
        );
        Ok(())
    }

    #[test]
    fn sorted_dict_gte_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Gte)?;
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([true, false, true, false, true, true])
        );
        Ok(())
    }

    #[test]
    fn sorted_dict_lte_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Lte)?;
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([false, true, true, true, false, true])
        );
        Ok(())
    }

    #[test]
    fn sorted_dict_gt_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let scalar = ConstantArray::new(2i32, 6).into_array();
        let r = cmp(dict, scalar, Operator::Gt)?;
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([true, false, false, false, true, false])
        );
        Ok(())
    }

    #[test]
    fn sorted_dict_eq_string() -> VortexResult<()> {
        let arr = VarBinArray::from_iter(
            [
                Some("zeta"),
                Some("alpha"),
                Some("mu"),
                Some("alpha"),
                Some("zeta"),
            ],
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
            [
                Some("zeta"),
                Some("alpha"),
                Some("mu"),
                Some("alpha"),
                Some("zeta"),
            ],
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
