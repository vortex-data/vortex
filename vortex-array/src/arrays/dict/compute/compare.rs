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
        // Sorted-values fast path: emit a code-domain compare directly.
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

            return Ok(Some(result.execute::<Canonical>(ctx)?.into_array()));
        }

        Ok(None)
    }
}

/// Sorted-values fast path for comparing a dict to a scalar constant.
///
/// Resolves the predicate `value OP scalar` to a code-domain integer comparison and emits
/// that compare directly against the codes child. Skips the `take(bool_values, codes)` step
/// the plain path needs.
///
/// Returns `None` if the values encoding doesn't support a typed linear scan (caller falls
/// back to the existing path).
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

    let Some(bounds) = scan_sorted_bounds(&values, scalar)? else {
        return Ok(None);
    };

    let const_bool = |b: bool| -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(Scalar::bool(b, result_nullability), codes_len).into_array())
    };

    match operator {
        CompareOperator::Eq => match bounds.found {
            Some(i) => code_cmp(&codes, i, Operator::Eq, ctx).map(Some),
            None => const_bool(false).map(Some),
        },
        CompareOperator::NotEq => match bounds.found {
            Some(i) => code_cmp(&codes, i, Operator::NotEq, ctx).map(Some),
            None => {
                if matches!(result_nullability, Nullability::Nullable) {
                    // Need null preservation through the existing path.
                    Ok(None)
                } else {
                    const_bool(true).map(Some)
                }
            }
        },
        // value < scalar  iff  code < left  (left = first idx where value >= scalar)
        CompareOperator::Lt => emit_lt(&codes, bounds.left, dict_len, result_nullability, ctx),
        // value <= scalar iff  code < right (right = first idx where value > scalar)
        CompareOperator::Lte => emit_lt(&codes, bounds.right, dict_len, result_nullability, ctx),
        // value > scalar  iff  code >= right
        CompareOperator::Gt => emit_gte(&codes, bounds.right, dict_len, result_nullability, ctx),
        // value >= scalar iff  code >= left
        CompareOperator::Gte => emit_gte(&codes, bounds.left, dict_len, result_nullability, ctx),
    }
}

fn emit_lt(
    codes: &ArrayRef,
    bound: usize,
    dict_len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if bound == 0 {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(false, nullability), codes.len()).into_array(),
        ));
    }
    if bound >= dict_len && !nullability.is_nullable() {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(true, nullability), codes.len()).into_array(),
        ));
    }
    code_cmp(codes, bound, Operator::Lt, ctx).map(Some)
}

fn emit_gte(
    codes: &ArrayRef,
    bound: usize,
    dict_len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if bound == 0 && !nullability.is_nullable() {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(true, nullability), codes.len()).into_array(),
        ));
    }
    if bound >= dict_len {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(false, nullability), codes.len()).into_array(),
        ));
    }
    code_cmp(codes, bound, Operator::Gte, ctx).map(Some)
}

/// Run a code-domain compare against an integer threshold. Calls `execute_compare`
/// directly, skipping the ScalarFnArray/executor scaffolding so the call drops straight
/// into Arrow's vectorized primitive cmp kernel.
pub(crate) fn code_cmp(
    codes: &ArrayRef,
    threshold: usize,
    op: Operator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    use crate::scalar_fn::fns::binary::execute_compare;
    let threshold_scalar = code_threshold_scalar(codes, threshold)?;
    let len = codes.len();
    let threshold_arr = ConstantArray::new(threshold_scalar, len).into_array();
    let compare_op = match op {
        Operator::Eq => CompareOperator::Eq,
        Operator::NotEq => CompareOperator::NotEq,
        Operator::Lt => CompareOperator::Lt,
        Operator::Lte => CompareOperator::Lte,
        Operator::Gt => CompareOperator::Gt,
        Operator::Gte => CompareOperator::Gte,
        _ => vortex_error::vortex_bail!("code_cmp called with non-comparison op {op:?}"),
    };
    execute_compare(codes, &threshold_arr, compare_op, ctx)
}

/// Build a `Scalar` of the codes' ptype holding `idx`. `idx` is in `0..=dict_len` and
/// `dict_len` already fits in the codes ptype by DictArray invariant.
fn code_threshold_scalar(codes: &ArrayRef, idx: usize) -> VortexResult<Scalar> {
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
