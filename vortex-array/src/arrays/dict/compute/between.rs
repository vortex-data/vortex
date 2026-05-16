// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sorted-values fast path for `BETWEEN` lives in `rules.rs` as a reduce rule that fires
//! before the value push-down. This module provides the AST builders that the rule uses.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::compute::compare::emit_code_cmp;
use crate::arrays::dict::compute::compare::scan_sorted_bounds;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::operators::Operator;

/// Reduce-rule entry point: emit the AST for a sorted-dict BETWEEN without running the
/// executor. Returns `None` if the typed scan doesn't apply (caller falls back to the
/// value push-down rule).
pub(crate) fn reduce_sorted_between(
    array: ArrayView<'_, Dict>,
    lower_scalar: &Scalar,
    upper_scalar: &Scalar,
    options: &BetweenOptions,
) -> VortexResult<Option<ArrayRef>> {
    if array.values().dtype().is_nullable() {
        return Ok(None);
    }

    let codes = array.codes().clone();
    let codes_len = codes.len();
    let nullability = codes.dtype().nullability();
    let values = array.values().clone();
    let dict_len = values.len();

    let Some(lower_bounds) = scan_sorted_bounds(&values, lower_scalar)? else {
        return Ok(None);
    };
    let Some(upper_bounds) = scan_sorted_bounds(&values, upper_scalar)? else {
        return Ok(None);
    };
    let code_lo = if options.lower_strict.is_strict() {
        lower_bounds.right
    } else {
        lower_bounds.left
    };
    let code_hi = if options.upper_strict.is_strict() {
        upper_bounds.left
    } else {
        upper_bounds.right
    };

    if code_lo >= code_hi {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(false, nullability), codes_len).into_array(),
        ));
    }
    if code_lo == 0 && code_hi >= dict_len && !nullability.is_nullable() {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(true, nullability), codes_len).into_array(),
        ));
    }
    if code_lo == 0 {
        return Ok(Some(emit_code_cmp(&codes, code_hi, Operator::Lt)?));
    }
    if code_hi >= dict_len {
        return Ok(Some(emit_code_cmp(&codes, code_lo, Operator::Gte)?));
    }

    let lo_cmp = emit_code_cmp(&codes, code_lo, Operator::Gte)?;
    let hi_cmp = emit_code_cmp(&codes, code_hi, Operator::Lt)?;
    Ok(Some(lo_cmp.binary(hi_cmp, Operator::And)?))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::builders::dict::dict_encode_sorted;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::between;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison::NonStrict;
    use crate::scalar_fn::fns::between::StrictComparison::Strict;

    fn apply_between(
        dict: ArrayRef,
        lo: i32,
        hi: i32,
        lower_strict: bool,
        upper_strict: bool,
    ) -> VortexResult<ArrayRef> {
        let opts = BetweenOptions {
            lower_strict: if lower_strict { Strict } else { NonStrict },
            upper_strict: if upper_strict { Strict } else { NonStrict },
        };
        let expr = between(root(), lit(lo), lit(hi), opts);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        dict.apply(&expr)?
            .execute::<crate::Canonical>(&mut ctx)
            .map(Into::into)
    }

    #[test]
    fn between_inclusive_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 5, 2, 4, 1, 5].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let r = apply_between(dict, 2, 4, false, false)?;
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([true, false, false, true, true, false, false])
        );
        Ok(())
    }

    #[test]
    fn between_exclusive_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 5, 2, 4, 1, 5].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let r = apply_between(dict, 2, 5, true, true)?;
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([true, false, false, false, true, false, false])
        );
        Ok(())
    }

    #[test]
    fn between_empty_range_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 5, 2, 4].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let r = apply_between(dict, 10, 20, false, false)?;
        assert_arrays_eq!(r, BoolArray::from_iter([false, false, false, false, false]));
        Ok(())
    }

    #[test]
    fn between_inclusive_string() -> VortexResult<()> {
        let arr = VarBinArray::from_iter(
            [
                Some("zeta"),
                Some("alpha"),
                Some("mu"),
                Some("alpha"),
                Some("kappa"),
                Some("zeta"),
            ],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let opts = BetweenOptions {
            lower_strict: NonStrict,
            upper_strict: NonStrict,
        };
        let expr = between(root(), lit("alpha"), lit("mu"), opts);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let r = dict
            .apply(&expr)?
            .execute::<crate::Canonical>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([false, true, true, true, true, false])
        );
        Ok(())
    }
}
