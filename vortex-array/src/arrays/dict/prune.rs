// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pruning unreferenced values from a [`DictArray`].
//!
//! When a [`DictArray`] survives selective compute (filter, take, …) only the codes change; the
//! values buffer is preserved unchanged. After cascading filters it is common for the codes to
//! reference only a small fraction of the values, but consumers that materialise the dictionary
//! (Arrow conversion, the DuckDB exporter, …) still pay to decompress and copy every value.
//!
//! [`maybe_prune_unreferenced_values`] discovers the referenced positions by walking the codes
//! once (without ever materialising a full `values.len()`-sized buffer), and — if the savings
//! exceed a threshold — returns a smaller equivalent [`DictArray`] whose values contain only
//! the referenced entries and whose codes are remapped accordingly. The remapped dict is
//! marked with `all_values_referenced = true` so downstream consumers can skip the same work.

use std::collections::BTreeSet;

use num_traits::AsPrimitive;
use num_traits::FromPrimitive;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use super::DictArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Minimum number of values before pruning is even considered.
///
/// The prune adds a fixed cost on top of any existing export: scanning the codes to find unique
/// referenced positions, taking those positions from the values, canonicalising the result, and
/// re-validating the new dict. For small dicts the baseline export already touches each value in
/// near-O(values.len()) time, so the overhead can dominate. We've measured the break-even point
/// to be in the tens of thousands of values for varbinview-shaped exports; below that, pruning
/// reliably regresses end-to-end wall time.
pub const PRUNE_DICT_MIN_VALUES: usize = 32 * 1024;

/// Minimum fraction of unreferenced values required to commit to a prune. If the savings are
/// smaller than this we keep the original dict so callers retain any cross-chunk sharing of the
/// values buffer.
pub const PRUNE_DICT_MIN_SAVINGS_RATIO: f64 = 0.5;

/// Inspect `array` and, if the codes leave a sizable fraction of `values` unreferenced, return a
/// compact [`DictArray`] whose values contain only the referenced entries.
///
/// Decisions:
///
/// 1. **Cheap rejections.** Skip when the values buffer is small, when the dict is already
///    known to be fully referenced, or when the values is a `Constant` (which has its own
///    dedicated exporter path).
/// 2. **One-pass discovery.** Walk the codes once, accumulating unique referenced positions in
///    a `BTreeSet`. Bail out early once the set grows past `(1 - threshold) * values.len()`
///    entries: at that point the savings ratio can no longer clear the threshold.
/// 3. **Commit.** Take the referenced values into a new array; remap codes through a binary
///    search against the sorted referenced positions; stamp `all_values_referenced = true`.
///
/// Returns `Ok(None)` to mean "no change worth making"; callers should keep the original array.
pub fn maybe_prune_unreferenced_values(
    array: &DictArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<DictArray>> {
    let values_len = array.values().len();

    // Cheap rejections.
    //
    // `has_all_values_referenced()` is the dense fast-path: `dict_encode` sets it; `filter` /
    // `take` / `slice` correctly clear it (they can orphan values).
    if values_len < PRUNE_DICT_MIN_VALUES || array.has_all_values_referenced() {
        return Ok(None);
    }
    // A Constant values dict has its own dedicated exporter path; pruning would just create a
    // shorter Constant, so let the caller deal with it directly.
    if array.values().is::<Constant>() {
        return Ok(None);
    }

    // Walk the codes once. Stop early if we collect enough unique references to make pruning
    // unprofitable (i.e., the savings ratio can't clear the threshold).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "threshold is a ratio in [0,1] and values_len fits in f64 for any sane dict"
    )]
    let cap_unique = ((values_len as f64) * (1.0 - PRUNE_DICT_MIN_SAVINGS_RATIO)) as usize;
    let referenced_positions = match collect_referenced_positions(array, ctx, cap_unique)? {
        Some(positions) => positions,
        None => return Ok(None),
    };
    debug_assert!(referenced_positions.len() <= cap_unique);

    // Degenerate: nothing referenced. Caller already has fast-paths for AllInvalid codes.
    if referenced_positions.is_empty() {
        return Ok(None);
    }

    Some(prune_with_positions(array, referenced_positions, ctx)).transpose()
}

/// Scan codes (host-resident only) and return the sorted list of unique referenced value
/// positions. Returns `Ok(None)` if the unique referenced count exceeds `cap_unique` — the
/// caller's signal that pruning isn't worth it.
fn collect_referenced_positions(
    array: &DictArray,
    ctx: &mut ExecutionCtx,
    cap_unique: usize,
) -> VortexResult<Option<Vec<u32>>> {
    let codes = array.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let codes_validity = codes
        .validity()?
        .execute_mask(codes.as_array().len(), ctx)?;

    // BTreeSet so we get the sorted output for free. For ~1000s of unique values this is fast;
    // we cap the size early so it can't blow up on dense dicts.
    let mut set = BTreeSet::<u32>::new();

    let overflowed = match codes_validity.bit_buffer() {
        AllOr::All => collect_all(&codes, &mut set, cap_unique),
        AllOr::None => false,
        AllOr::Some(mask) => collect_masked(&codes, mask, &mut set, cap_unique),
    };
    if overflowed {
        return Ok(None);
    }
    Ok(Some(set.into_iter().collect()))
}

fn collect_all(codes: &PrimitiveArray, set: &mut BTreeSet<u32>, cap: usize) -> bool {
    match_each_integer_ptype!(codes.ptype(), |C| {
        for &c in codes.as_slice::<C>() {
            let pos: u32 = AsPrimitive::<u32>::as_(c);
            if set.insert(pos) && set.len() > cap {
                return true;
            }
        }
    });
    false
}

fn collect_masked(
    codes: &PrimitiveArray,
    mask: &vortex_buffer::BitBuffer,
    set: &mut BTreeSet<u32>,
    cap: usize,
) -> bool {
    match_each_integer_ptype!(codes.ptype(), |C| {
        let slice = codes.as_slice::<C>();
        for idx in mask.set_indices() {
            let pos: u32 = AsPrimitive::<u32>::as_(slice[idx]);
            if set.insert(pos) && set.len() > cap {
                return true;
            }
        }
    });
    false
}

/// Apply a precomputed sorted list of referenced value positions: take the referenced values,
/// remap the codes via binary search, and stamp `all_values_referenced = true`.
fn prune_with_positions(
    array: &DictArray,
    referenced_positions: Vec<u32>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    debug_assert!(!referenced_positions.is_empty());

    // Materialise the codes on the host, then remap them through the sorted-positions list
    // while preserving validity.
    let codes = array.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let codes_validity = codes.validity()?;
    let codes_array = match_each_integer_ptype!(codes.ptype(), |C| {
        remap_codes::<C>(codes.as_slice::<C>(), &referenced_positions, codes_validity).into_array()
    });

    // Take the referenced values. `take` returns a lazy `DictArray(take_indices, values)`
    // wrapper that we must execute to its canonical form — otherwise the duckdb exporter would
    // see a `Dict<Dict<…>>` and recurse into pruning forever.
    let take_indices = PrimitiveArray::from_iter(referenced_positions.iter().copied()).into_array();
    let new_values = array
        .values()
        .take(take_indices)?
        .execute::<crate::Canonical>(ctx)?
        .into_array();

    // SAFETY: We just enumerated every referenced value, so the resulting dict is fully
    // referenced.
    Ok(
        unsafe {
            DictArray::new_unchecked(codes_array, new_values).set_all_values_referenced(true)
        },
    )
}

fn remap_codes<C: IntegerPType + AsPrimitive<u32> + FromPrimitive>(
    codes: &[C],
    referenced_positions: &[u32],
    validity: Validity,
) -> PrimitiveArray {
    // Each valid code's underlying value index appears in `referenced_positions`. Its new code
    // is the index in this sorted list. For invalid codes the original index may not be in
    // the list — binary_search returns the insertion point, which we clamp to keep the result
    // in-range; validity hides the value anyway.
    let last_valid_new_idx =
        u32::try_from(referenced_positions.len().saturating_sub(1)).unwrap_or(u32::MAX);
    let mut out = BufferMut::<C>::with_capacity(codes.len());
    for &c in codes {
        let original: u32 = c.as_();
        let new = match referenced_positions.binary_search(&original) {
            Ok(pos) => u32::try_from(pos).unwrap_or(u32::MAX),
            Err(pos) => u32::try_from(pos)
                .unwrap_or(u32::MAX)
                .min(last_valid_new_idx),
        };
        // The new code is always <= original code (binary_search position <= original), so it
        // fits in the original code's integer width.
        out.push(<C as FromPrimitive>::from_u32(new).unwrap_or_else(|| {
            debug_assert!(false, "new code {new} does not fit in original width");
            C::zero()
        }));
    }
    PrimitiveArray::new(out.freeze(), validity)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;

    fn canon(array: ArrayRef) -> PrimitiveArray {
        array
            .execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    }

    #[test]
    fn returns_none_for_small_dict() -> VortexResult<()> {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 0, 1].into_array(),
            buffer![10i64, 20].into_array(),
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_prune_unreferenced_values(&dict, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn returns_none_when_savings_too_small() -> VortexResult<()> {
        // Values [0..40_960), codes reference [0..30_000) (~27% unreferenced — below threshold).
        let values = PrimitiveArray::from_iter(0i64..40_960).into_array();
        let codes: Vec<u32> = (0..30_000).map(|i| i as u32).collect();
        let dict = DictArray::try_new(PrimitiveArray::from_iter(codes).into_array(), values)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_prune_unreferenced_values(&dict, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn prunes_when_codes_are_sparse() -> VortexResult<()> {
        // Values [0..40_960), codes reference [0..100) (~99% unreferenced).
        let values = PrimitiveArray::from_iter(0i64..40_960).into_array();
        let codes: Vec<u32> = (0..2048).map(|i| (i % 100) as u32).collect();
        let dict = DictArray::try_new(
            PrimitiveArray::from_iter(codes.clone()).into_array(),
            values,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let pruned = maybe_prune_unreferenced_values(&dict, &mut ctx)?
            .expect("expected a prune with sparse codes");
        assert_eq!(pruned.values().len(), 100);
        assert!(pruned.has_all_values_referenced());

        // Each code i % 100 must round-trip back to the same value through the pruned dict.
        let pruned_codes = canon(pruned.codes().clone());
        let pruned_values = canon(pruned.values().clone());
        for (i, &orig_code) in codes.iter().enumerate() {
            let new_code: u32 = pruned_codes.as_slice::<u32>()[i];
            let value: i64 = pruned_values.as_slice::<i64>()[new_code as usize];
            assert_eq!(
                value, orig_code as i64,
                "round-trip mismatch at position {i}"
            );
        }
        Ok(())
    }

    #[test]
    fn preserves_validity_when_codes_are_nullable() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0i64..40_960).into_array();
        // Codes: about 90% reference only [0..50), every 13th is null.
        let codes_opt: Vec<Option<u32>> = (0..2048)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some((i % 50) as u32)
                }
            })
            .collect();
        let codes = PrimitiveArray::from_option_iter(codes_opt.clone()).into_array();
        let dict = DictArray::try_new(codes, values)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let pruned = maybe_prune_unreferenced_values(&dict, &mut ctx)?
            .expect("expected a prune with nullable sparse codes");
        assert_eq!(pruned.values().len(), 50);

        // Validity of the pruned dict's codes must match the input's null pattern.
        let pruned_codes_view = pruned
            .codes()
            .as_opt::<Primitive>()
            .ok_or_else(|| vortex_error::vortex_err!("pruned codes must be primitive"))?;
        let mask = pruned_codes_view.validity()?.execute_mask(2048, &mut ctx)?;
        for (i, expected) in codes_opt.iter().enumerate() {
            assert_eq!(
                mask.value(i),
                expected.is_some(),
                "validity mismatch at {i}"
            );
        }
        Ok(())
    }

    #[test]
    fn round_trips_against_canonical() -> VortexResult<()> {
        // Set up a sparse dict and verify the pruned variant canonicalises identically.
        let values = PrimitiveArray::from_iter(0i64..40_960).into_array();
        let codes: Vec<u32> = (0..2048).map(|i| ((i * 17) % 64) as u32).collect();
        let dict = DictArray::try_new(PrimitiveArray::from_iter(codes).into_array(), values)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let pruned = maybe_prune_unreferenced_values(&dict, &mut ctx)?
            .expect("expected a prune for round-trip test");

        let original_canonical = canon(dict.into_array());
        let pruned_canonical = canon(pruned.into_array());
        assert_arrays_eq!(original_canonical, pruned_canonical);
        Ok(())
    }
}
