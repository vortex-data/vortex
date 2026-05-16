// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sorted-values fast path for `BETWEEN`.
//!
//! For sorted-values dicts, `value BETWEEN lo AND hi` translates to a contiguous range of
//! dictionary slots. Resolve each bound via one binary search on the values array, build a
//! small `BoolArray` of length `dict_len` representing the predicate per slot, wrap as a
//! `DictArray<codes, bool_values>`, and canonicalize. This is the same `take(bool, codes)`
//! pipeline as the existing dict-compare path, but the values-side comparison is replaced
//! by O(log dict_len) instead of O(dict_len * compare_cost) per bound.

use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenKernel;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

impl BetweenKernel for Dict {
    fn between(
        array: ArrayView<'_, Dict>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower_scalar), Some(upper_scalar)) = (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };

        if lower_scalar.is_null() || upper_scalar.is_null() {
            return Ok(None);
        }

        if !array.has_sorted_values() || array.values().dtype().is_nullable() {
            return Ok(None);
        }

        let codes = array.codes().clone();
        let codes_len = codes.len();
        let nullability = codes.dtype().nullability();
        let values = array.values().clone();
        let dict_len = values.len();

        let code_lo = if options.lower_strict.is_strict() {
            values
                .search_sorted(&lower_scalar, SearchSortedSide::Right)?
                .to_index()
        } else {
            values
                .search_sorted(&lower_scalar, SearchSortedSide::Left)?
                .to_index()
        };
        let code_hi = if options.upper_strict.is_strict() {
            values
                .search_sorted(&upper_scalar, SearchSortedSide::Left)?
                .to_index()
        } else {
            values
                .search_sorted(&upper_scalar, SearchSortedSide::Right)?
                .to_index()
        };

        if code_lo >= code_hi {
            return Ok(Some(
                ConstantArray::new(Scalar::bool(false, nullability), codes_len).into_array(),
            ));
        }

        // Build a non-nullable bool array of length dict_len:
        //   slot i == true  iff  code_lo <= i < code_hi
        let mut bb = BitBufferMut::with_capacity(dict_len);
        bb.append_n(false, code_lo);
        bb.append_n(true, code_hi - code_lo);
        bb.append_n(false, dict_len - code_hi);
        let bool_values = BoolArray::new(bb.freeze(), Validity::NonNullable).into_array();

        // SAFETY: bool_values len == dict_len; codes index into [0, dict_len) by DictArray
        // invariant.
        let dict = unsafe { DictArray::new_unchecked(codes, bool_values).into_array() };
        Ok(Some(dict.execute::<Canonical>(ctx)?.into_array()))
    }
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
        dict.apply(&expr)?.execute::<crate::Canonical>(&mut ctx).map(Into::into)
    }

    #[test]
    fn between_inclusive_primitive() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 5, 2, 4, 1, 5].into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        // 2 <= x <= 4
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
        // 2 < x < 5
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
        // 10 <= x <= 20  → all false
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
        let r = dict.apply(&expr)?.execute::<crate::Canonical>(&mut ctx)?.into_array();
        // "alpha" <= x <= "mu" → alpha, mu, alpha, kappa
        assert_arrays_eq!(
            r,
            BoolArray::from_iter([false, true, true, true, true, false])
        );
        Ok(())
    }
}
