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
///
/// TODO(sorted-column): a sorted-values dict still has unsorted codes, so a middle-range
/// `lt` predicate has to run a SIMD compare over every code (~22 µs at 100K, u16).
/// `benches/dict_sorted_pushdowns.rs::cmp_*` shows that:
///   - the SIMD bitmap pack is already optimal — faster than `Vec<bool>`, faster even
///     than a scalar count;
///   - the only way to beat it is to skip per-row evaluation, which requires the *codes*
///     to be sorted (i.e. a fully sorted column, not just a sorted-values dict). In that
///     case the predicate collapses to `Mask::from_slices(N, [(0, k)])` at ~350 ns,
///     a ~64x further speedup.
/// Reuses the same `partition_point` we already do in `scan_primitive_from`; gated on a
/// new `has_sorted_codes()` (or equivalent layout-level marker).
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
        // value < scalar  iff  code < left   (left = first idx where value >= scalar)
        CompareOperator::Lt => {
            emit_bounded_cmp(&codes, bounds.left, dict_len, Operator::Lt, result_nullability, codes_len)
        }
        // value <= scalar iff  code < right  (right = first idx where value > scalar)
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

/// Emit a code-domain `code OP bound` compare, folding to a `ConstantArray<Bool>` when
/// `bound` is at an extreme (`0` or `dict_len`) so `Mask::execute` collapses the result
/// to `AllTrue`/`AllFalse` in O(1). `op` must be `Lt` or `Gte`; the symmetric values for
/// each boundary are determined by which direction the comparison extends.
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
    // For Lt: `code < 0` is always false; `code < dict_len` is always true.
    // For Gte: `code >= 0` is always true; `code >= dict_len` is always false.
    let (at_zero, at_top) = match op {
        Operator::Lt => (false, true),
        Operator::Gte => (true, false),
        _ => vortex_error::vortex_panic!("emit_bounded_cmp expects Lt or Gte, got {op:?}"),
    };
    if bound == 0 {
        // The `true` arm requires non-nullable codes; otherwise we'd lose null
        // propagation and need to fall back to the existing path.
        if !at_zero || !nullability.is_nullable() {
            return Ok(Some(const_bool(at_zero)));
        }
    }
    if bound >= dict_len {
        if !at_top || !nullability.is_nullable() {
            return Ok(Some(const_bool(at_top)));
        }
    }
    Ok(Some(emit_code_cmp(codes, bound, op)?))
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
    use crate::match_each_integer_ptype;
    let nullability = codes.dtype().nullability();
    let DType::Primitive(ptype, _) = codes.dtype() else {
        vortex_error::vortex_bail!("dict codes have unexpected dtype {}", codes.dtype());
    };
    Ok(match_each_integer_ptype!(ptype, |T| {
        // SAFETY: dict_len fits in the codes ptype by DictArray invariant; idx <= dict_len.
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
        let (Some(lo_needle), Some(hi_needle)) =
            (scalar_as_bytes(lower), scalar_as_bytes(upper))
        else {
            return Ok(None);
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

/// Extract a byte-slice view of a Utf8 / Binary scalar. Returns `None` for any other
/// dtype or null value. Shared between the single-bound and dual-bound scans.
fn scalar_as_bytes(scalar: &Scalar) -> Option<&[u8]> {
    use crate::dtype::DType;
    match scalar.dtype() {
        DType::Utf8(_) => scalar.as_utf8_opt().and_then(|v| v.value()).map(|s| s.as_bytes()),
        DType::Binary(_) => scalar.as_binary_opt().and_then(|v| v.value()).map(|b| b.as_slice()),
        _ => None,
    }
}

/// Two-phase scan. Phase 1 walks `slice` to find `lo` bounds; since values are sorted
/// ascending and `hi >= lo`, phase 2 resumes from where phase 1 stopped. Each element is
/// compared against exactly one needle. Special-cases `lo == hi` so both bounds match.
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

/// Two-phase dual scan for byte slices. Materializes the view iterator once (cheap:
/// dict_len <= u16::MAX, view resolution is essentially free) then runs the shared
/// `scan_bytes_from` helper twice: phase 1 finds lo bounds, phase 2 resumes from there
/// to find hi bounds. Each element is compared against exactly one needle.
fn scan_bytes_dual<'a>(
    it: &mut dyn Iterator<Item = Option<&'a [u8]>>,
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
        let Some(needle) = scalar_as_bytes(scalar) else {
            return Ok(None);
        };
        let bounds = vbv
            .into_owned()
            .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| scan_bytes(it, needle));
        return Ok(Some(bounds));
    }
    Ok(None)
}

fn scan_primitive<T: crate::dtype::NativePType>(slice: &[T], needle: T) -> SortedBounds {
    let (bounds, _) = scan_primitive_from(slice, 0, needle);
    bounds
}

/// Walk `slice[start..]` and locate `(left, right, found)` boundaries for `needle`,
/// stopping at the first element greater than `needle`. Also returns the index where
/// the scan exited (`slice.len()` if it reached the end without finding `Greater`).
///
/// Shared by single-bound `scan_primitive` and dual-bound `scan_primitive_dual` so the
/// scan loop is implemented once.
fn scan_primitive_from<T: crate::dtype::NativePType>(
    slice: &[T],
    start: usize,
    needle: T,
) -> (SortedBounds, usize) {
    use std::cmp::Ordering::*;
    let n = slice.len();
    let mut left: Option<usize> = None;
    let mut right: Option<usize> = None;
    let mut found: Option<usize> = None;
    let mut exit = n;
    let mut i = start;
    while i < n {
        // SAFETY: i < n.
        let v = unsafe { *slice.get_unchecked(i) };
        match v.total_compare(needle) {
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
                exit = i;
                break;
            }
        }
        i += 1;
    }
    (
        SortedBounds {
            left: left.unwrap_or(n),
            right: right.unwrap_or(n),
            found,
        },
        exit,
    )
}

fn scan_bytes<'a>(
    it: &mut dyn Iterator<Item = Option<&'a [u8]>>,
    needle: &[u8],
) -> SortedBounds {
    // The single-bound case still materialises so we share scan_bytes_from with the
    // dual-bound case. The Vec<Option<&[u8]>> is small (dict_len <= u16::MAX).
    let items: Vec<Option<&[u8]>> = it.collect();
    let (bounds, _) = scan_bytes_from(&items, 0, needle);
    bounds
}

/// Walk `items[start..]` and locate `(left, right, found)` boundaries for `needle`,
/// stopping at the first byte slice greater than it. Returns the exit index (the
/// position where the scan stopped, or `items.len()` if none).
fn scan_bytes_from(items: &[Option<&[u8]>], start: usize, needle: &[u8]) -> (SortedBounds, usize) {
    use std::cmp::Ordering::*;
    let n = items.len();
    let mut left: Option<usize> = None;
    let mut right: Option<usize> = None;
    let mut found: Option<usize> = None;
    let mut exit = n;
    for (i, opt) in items.iter().copied().enumerate().skip(start) {
        let cmp = match opt {
            None => Less, // nulls sort first
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
                exit = i;
                break;
            }
        }
    }
    (
        SortedBounds {
            left: left.unwrap_or(n),
            right: right.unwrap_or(n),
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
