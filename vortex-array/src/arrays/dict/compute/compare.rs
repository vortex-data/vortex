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
        // Note: the sorted-values fast path lives in `DictionarySortedCompareRule` in
        // `rules.rs`, which fires before `DictionaryScalarFnValuesPushDownRule` to rewrite
        // the predicate as a codes-domain compare. By the time CompareKernel is reached, we
        // are on the values-side compare path that the push-down rule produced — fall
        // through to the existing logic.

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

            return Ok(Some(result.execute::<Canonical>(ctx)?.into_array()));
        }

        Ok(None)
    }
}

/// Reduce-rule entry point: emit the AST for a sorted-dict compare without running the
/// executor.
///
/// Returns:
/// - `Ok(Some(ScalarFnArray(Binary, [codes, const])))` — the codes-domain compare. This
///   always saves the value-side scan, but for middle-range predicates it ties or
///   slightly loses against the take-based plain path. We still emit it because the
///   real win on real workloads is when downstream encodings (FoR / bit-packed) can
///   answer the codes-domain compare on the compressed buffer.
/// - `Ok(Some(ConstantArray))` — short-circuit when the predicate evaluates to a constant
///   (no match / boundary out of dict range). Huge win because the canonical Mask becomes
///   AllTrue / AllFalse in O(1).
/// - `Ok(None)` if the typed scan doesn't apply (caller falls back to the value push-down).
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

    // Strategy: always emit a code-domain rewrite. With the Vortex-native primitive
    // CompareKernel (chunked 8-bit packing), cmp(codes, threshold) is now 2-3× faster
    // than take(bool[dict_len], codes) past L1 — the SIMD-friendly sequential cmp
    // beats the gather's dependent loads.  When the boundary collapses to an extreme
    // the result becomes a ConstantArray that Mask::execute folds to AllTrue/AllFalse
    // in O(1) — 17-57× wins on no-match equality.
    match operator {
        CompareOperator::Eq => match bounds.found {
            Some(i) => Ok(Some(emit_code_cmp(&codes, i, Operator::Eq)?)),
            None => Ok(Some(const_bool(false))),
        },
        CompareOperator::NotEq => match bounds.found {
            Some(i) => Ok(Some(emit_code_cmp(&codes, i, Operator::NotEq)?)),
            None => {
                if matches!(result_nullability, Nullability::Nullable) {
                    Ok(None)
                } else {
                    Ok(Some(const_bool(true)))
                }
            }
        },
        // value < scalar  iff  code < left  (left = first idx where value >= scalar)
        CompareOperator::Lt => emit_lt_or_cmp(
            &codes,
            bounds.left,
            dict_len,
            result_nullability,
            codes_len,
        ),
        // value <= scalar iff  code < right
        CompareOperator::Lte => emit_lt_or_cmp(
            &codes,
            bounds.right,
            dict_len,
            result_nullability,
            codes_len,
        ),
        // value > scalar  iff  code >= right
        CompareOperator::Gt => emit_gte_or_cmp(
            &codes,
            bounds.right,
            dict_len,
            result_nullability,
            codes_len,
        ),
        // value >= scalar iff  code >= left
        CompareOperator::Gte => emit_gte_or_cmp(
            &codes,
            bounds.left,
            dict_len,
            result_nullability,
            codes_len,
        ),
    }
}

/// `result = (code < bound)`. Folds to a constant when `bound` is at an extreme;
/// otherwise emits a primitive `code < bound` compare that the new Vortex-native
/// primitive cmp kernel runs as a single SIMD pass with bit-packed output.
fn emit_lt_or_cmp(
    codes: &ArrayRef,
    bound: usize,
    dict_len: usize,
    nullability: Nullability,
    codes_len: usize,
) -> VortexResult<Option<ArrayRef>> {
    let const_bool = |b: bool| -> ArrayRef {
        ConstantArray::new(Scalar::bool(b, nullability), codes_len).into_array()
    };
    if bound == 0 {
        return Ok(Some(const_bool(false)));
    }
    if bound >= dict_len && !nullability.is_nullable() {
        return Ok(Some(const_bool(true)));
    }
    Ok(Some(emit_code_cmp(codes, bound, Operator::Lt)?))
}

/// `result = (code >= bound)`. Folds to a constant when `bound` is at an extreme;
/// otherwise emits a primitive `code >= bound` compare.
fn emit_gte_or_cmp(
    codes: &ArrayRef,
    bound: usize,
    dict_len: usize,
    nullability: Nullability,
    codes_len: usize,
) -> VortexResult<Option<ArrayRef>> {
    let const_bool = |b: bool| -> ArrayRef {
        ConstantArray::new(Scalar::bool(b, nullability), codes_len).into_array()
    };
    if bound == 0 && !nullability.is_nullable() {
        return Ok(Some(const_bool(true)));
    }
    if bound >= dict_len {
        return Ok(Some(const_bool(false)));
    }
    Ok(Some(emit_code_cmp(codes, bound, Operator::Gte)?))
}

/// Build a code-domain compare expression: `codes OP threshold`. The Vortex-native
/// `CompareKernel for Primitive` then runs this as a chunked SIMD compare with
/// bit-packed output (3× faster than the take_bool path past L1).
pub(crate) fn emit_code_cmp(
    codes: &ArrayRef,
    threshold: usize,
    op: Operator,
) -> VortexResult<ArrayRef> {
    let threshold_scalar = code_threshold_scalar(codes, threshold)?;
    let len = codes.len();
    let threshold_arr = ConstantArray::new(threshold_scalar, len).into_array();
    codes.clone().binary(threshold_arr, op)
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

/// Boundary indices on a sorted (ascending, non-nullable) values array.
#[derive(Debug)]
pub(crate) struct SortedBounds {
    /// First index `i` where `values[i] >= scalar`.
    pub left: usize,
    /// First index `i` where `values[i] > scalar`.
    pub right: usize,
    /// `Some(i)` iff `values[i] == scalar`; `None` if no exact match.
    pub found: Option<usize>,
}

/// Scan a sorted values array to find boundaries for `lower` AND `upper` in a single
/// pass. Used by the BETWEEN reduce rule, which would otherwise call `scan_sorted_bounds`
/// twice. Since values are sorted ascending, both needles can be resolved together: once
/// we pass `upper`, no later element can affect either bound.
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
        let lo_needle: &[u8] = match lower.dtype() {
            crate::dtype::DType::Utf8(_) => {
                let Some(s) = lower.as_utf8_opt().and_then(|v| v.value()) else {
                    return Ok(None);
                };
                s.as_bytes()
            }
            crate::dtype::DType::Binary(_) => {
                let Some(b) = lower.as_binary_opt().and_then(|v| v.value()) else {
                    return Ok(None);
                };
                b.as_slice()
            }
            _ => return Ok(None),
        };
        let hi_needle: &[u8] = match upper.dtype() {
            crate::dtype::DType::Utf8(_) => {
                let Some(s) = upper.as_utf8_opt().and_then(|v| v.value()) else {
                    return Ok(None);
                };
                s.as_bytes()
            }
            crate::dtype::DType::Binary(_) => {
                let Some(b) = upper.as_binary_opt().and_then(|v| v.value()) else {
                    return Ok(None);
                };
                b.as_slice()
            }
            _ => return Ok(None),
        };
        let bounds = vbv
            .into_owned()
            .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
                scan_bytes_dual(it, lo_needle, hi_needle)
            });
        return Ok(Some(bounds));
    }
    Ok(None)
}

fn scan_primitive_dual<T: crate::dtype::NativePType>(
    slice: &[T],
    lo: T,
    hi: T,
) -> (SortedBounds, SortedBounds) {
    use std::cmp::Ordering::*;
    let mut lo_left: Option<usize> = None;
    let mut lo_right: Option<usize> = None;
    let mut lo_found: Option<usize> = None;
    let mut hi_left: Option<usize> = None;
    let mut hi_right: Option<usize> = None;
    let mut hi_found: Option<usize> = None;

    for (i, &v) in slice.iter().enumerate() {
        if hi_right.is_none() {
            match v.total_compare(hi) {
                Less => {}
                Equal => {
                    if hi_left.is_none() {
                        hi_left = Some(i);
                        hi_found = Some(i);
                    }
                }
                Greater => {
                    if hi_left.is_none() {
                        hi_left = Some(i);
                    }
                    hi_right = Some(i);
                }
            }
        }
        if lo_right.is_none() {
            match v.total_compare(lo) {
                Less => {}
                Equal => {
                    if lo_left.is_none() {
                        lo_left = Some(i);
                        lo_found = Some(i);
                    }
                }
                Greater => {
                    if lo_left.is_none() {
                        lo_left = Some(i);
                    }
                    lo_right = Some(i);
                }
            }
        }
        if hi_right.is_some() {
            break;
        }
    }
    let n = slice.len();
    (
        SortedBounds {
            left: lo_left.unwrap_or(n),
            right: lo_right.unwrap_or(n),
            found: lo_found,
        },
        SortedBounds {
            left: hi_left.unwrap_or(n),
            right: hi_right.unwrap_or(n),
            found: hi_found,
        },
    )
}

fn scan_bytes_dual<'a>(
    it: &mut dyn Iterator<Item = Option<&'a [u8]>>,
    lo: &[u8],
    hi: &[u8],
) -> (SortedBounds, SortedBounds) {
    use std::cmp::Ordering::*;
    let mut lo_left: Option<usize> = None;
    let mut lo_right: Option<usize> = None;
    let mut lo_found: Option<usize> = None;
    let mut hi_left: Option<usize> = None;
    let mut hi_right: Option<usize> = None;
    let mut hi_found: Option<usize> = None;
    let mut n = 0usize;
    for (i, opt) in it.enumerate() {
        n = i + 1;
        let bytes = match opt {
            None => {
                continue;
            }
            Some(b) => b,
        };
        if hi_right.is_none() {
            match bytes.cmp(hi) {
                Less => {}
                Equal => {
                    if hi_left.is_none() {
                        hi_left = Some(i);
                        hi_found = Some(i);
                    }
                }
                Greater => {
                    if hi_left.is_none() {
                        hi_left = Some(i);
                    }
                    hi_right = Some(i);
                }
            }
        }
        if lo_right.is_none() {
            match bytes.cmp(lo) {
                Less => {}
                Equal => {
                    if lo_left.is_none() {
                        lo_left = Some(i);
                        lo_found = Some(i);
                    }
                }
                Greater => {
                    if lo_left.is_none() {
                        lo_left = Some(i);
                    }
                    lo_right = Some(i);
                }
            }
        }
        if hi_right.is_some() {
            break;
        }
    }
    (
        SortedBounds {
            left: lo_left.unwrap_or(n),
            right: lo_right.unwrap_or(n),
            found: lo_found,
        },
        SortedBounds {
            left: hi_left.unwrap_or(n),
            right: hi_right.unwrap_or(n),
            found: hi_found,
        },
    )
}

/// Linear scan of a sorted values array to find the (left, right, found) boundaries for
/// `scalar`. Dict size is bounded by `u16::MAX` so a typed linear scan is fast (memory-
/// resident, well-predicted branch); the scaffolding of binary_search + `IndexOrd<Scalar>`
/// is unwarranted at this scale.
///
/// Returns `None` if `values` isn't a canonical Primitive or VarBinView or the scalar can't
/// be converted to the matching native type.
pub(crate) fn scan_sorted_bounds(
    values: &ArrayRef,
    scalar: &Scalar,
) -> VortexResult<Option<SortedBounds>> {
    use crate::accessor::ArrayAccessor;
    use crate::arrays::Primitive;
    use crate::arrays::VarBinView;
    use crate::match_each_native_ptype;

    if let Some(prim) = values.as_opt::<Primitive>() {
        return Ok(Some(match_each_native_ptype!(prim.ptype(), |T| {
            let Ok(needle) = T::try_from(scalar) else {
                return Ok(None);
            };
            scan_primitive::<T>(prim.as_slice::<T>(), needle)
        })));
    }
    if let Some(vbv) = values.as_opt::<VarBinView>() {
        let needle: &[u8] = match scalar.dtype() {
            crate::dtype::DType::Utf8(_) => {
                let Some(s) = scalar.as_utf8_opt().and_then(|v| v.value()) else {
                    return Ok(None);
                };
                s.as_bytes()
            }
            crate::dtype::DType::Binary(_) => {
                let Some(b) = scalar.as_binary_opt().and_then(|v| v.value()) else {
                    return Ok(None);
                };
                b.as_slice()
            }
            _ => return Ok(None),
        };
        let bounds = vbv
            .into_owned()
            .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| scan_bytes(it, needle));
        return Ok(Some(bounds));
    }
    Ok(None)
}

fn scan_primitive<T: crate::dtype::NativePType>(slice: &[T], needle: T) -> SortedBounds {
    use std::cmp::Ordering::*;
    let mut left: Option<usize> = None;
    let mut right: Option<usize> = None;
    let mut found: Option<usize> = None;
    for (i, &v) in slice.iter().enumerate() {
        match v.total_compare(needle) {
            Less => {} // continue
            Equal => {
                if left.is_none() {
                    left = Some(i);
                    found = Some(i);
                }
            }
            Greater => {
                if left.is_none() {
                    left = Some(i);
                }
                right = Some(i);
                break;
            }
        }
    }
    SortedBounds {
        left: left.unwrap_or(slice.len()),
        right: right.unwrap_or(slice.len()),
        found,
    }
}

fn scan_bytes<'a>(
    it: &mut dyn Iterator<Item = Option<&'a [u8]>>,
    needle: &[u8],
) -> SortedBounds {
    use std::cmp::Ordering::*;
    let mut left: Option<usize> = None;
    let mut right: Option<usize> = None;
    let mut found: Option<usize> = None;
    let mut len = 0usize;
    for (i, opt) in it.enumerate() {
        len = i + 1;
        let cmp = match opt {
            None => Less, // nulls sort first; we early-rejected nullable values above, but be safe
            Some(b) => b.cmp(needle),
        };
        match cmp {
            Less => {}
            Equal => {
                if left.is_none() {
                    left = Some(i);
                    found = Some(i);
                }
            }
            Greater => {
                if left.is_none() {
                    left = Some(i);
                }
                right = Some(i);
                break;
            }
        }
    }
    // Drain the iterator length if we early-exited so callers don't get a misleading slice.
    SortedBounds {
        left: left.unwrap_or(len),
        right: right.unwrap_or(len),
        found,
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
