// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Trimming leading/trailing unreferenced elements from a [`ListArray`].
//!
//! [`ListArray`] guarantees monotonic offsets, so unreferenced elements can only appear in the
//! prefix `[0, offsets[0])` or suffix `[offsets[len], elements.len())` of the elements buffer.
//! After a slice or a fresh load from a file that retained surrounding data, both prefixes and
//! suffixes can be sizable. Consumers that materialise the elements (Arrow conversion, the
//! DuckDB exporter, …) still pay to decompress and copy every leading/trailing element when
//! they're never referenced by any list.
//!
//! [`maybe_trim_unreferenced_elements`] sniffs `offsets[0]` and `offsets[len]` and, if the
//! prefix+suffix exceeds a savings threshold, returns a trimmed [`ListArray`] whose elements
//! buffer is `elements.slice(offsets[0]..offsets[len])` and whose offsets are shifted by
//! `offsets[0]`.

use vortex_error::VortexResult;

use super::ListArray;
use super::ListArrayExt;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Minimum elements buffer length before pruning is considered, when elements are in a
/// canonical (host) encoding.
///
/// Below this size the original elements buffer is small enough that the baseline duckdb export
/// (zero-copy reference + a `Vector::with_capacity` allocation) is already cheap, and the trim
/// (offsets subtraction + slice) has fixed per-call overhead that doesn't pay back.
pub const PRUNE_LIST_MIN_ELEMENTS_CANONICAL: usize = 64 * 1024;

/// Minimum elements buffer length before pruning is considered, when elements are
/// **non-canonical** (e.g. `Dict`, `Fsst`, `RunEnd`, …).
///
/// Compressed elements pay per-position decompression at export time, so smaller buffers
/// benefit. Calibrated against the offsets-rewrite cost: at ≤16 KiB elements the
/// `O(num_lists)` binary subtract roughly matches the saved decompression; above ~32 KiB the
/// savings dominate reliably.
pub const PRUNE_LIST_MIN_ELEMENTS_COMPRESSED: usize = 32 * 1024;

/// Minimum fraction of `elements.len()` that must be unreferenced before we commit to a trim,
/// when elements are in a canonical encoding.
///
/// The trim does `O(num_lists)` work to rewrite the offsets via a binary subtract, plus a slice
/// of the elements (which is metadata-only). For canonical export the per-element cost is
/// roughly a `memcpy`, so we only fire when the prefix/suffix dominates the buffer — typically
/// after slicing a chunked list near a chunk boundary, or when loading a list column slice from
/// a file.
pub const PRUNE_LIST_MIN_SAVINGS_RATIO_CANONICAL: f64 = 0.97;

/// Minimum fraction of `elements.len()` that must be unreferenced before we commit to a trim,
/// when elements are compressed.
///
/// Compressed exports do per-position decompression. Even 50% prefix/suffix waste on a
/// dict-encoded varbin pays back: the offsets-rewrite is O(num_lists) (cheap), and the saved
/// decompression dominates.
pub const PRUNE_LIST_MIN_SAVINGS_RATIO_COMPRESSED: f64 = 0.50;

/// Inspect `array` and, if the prefix/suffix outside `[offsets[0], offsets[len])` exceeds the
/// savings threshold, return a trimmed [`ListArray`] whose elements contain only the referenced
/// positions.
///
/// Decisions:
///
/// 1. **Encoding-aware thresholds.** Pick `MIN_ELEMENTS` / `MIN_SAVINGS_RATIO` based on
///    whether `elements` is in a canonical encoding. Compressed children justify a much more
///    aggressive trim.
/// 2. **Cheap rejections.** Skip when the elements buffer is below the chosen `MIN_ELEMENTS`.
/// 3. **Endpoint sniff.** Read `offsets[0]` and `offsets[len]` (O(1) when offsets are host
///    primitive; otherwise we fall back to `execute_scalar`). Compute the trim savings.
/// 4. **Commit.** Slice the elements buffer and rewrite offsets by subtracting `offsets[0]`.
pub fn maybe_trim_unreferenced_elements(
    array: &ListArray,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ListArray>> {
    let elements_len = array.elements().len();
    let canonical_elements = array.elements().is_canonical();
    let (min_elements, min_savings) = if canonical_elements {
        (
            PRUNE_LIST_MIN_ELEMENTS_CANONICAL,
            PRUNE_LIST_MIN_SAVINGS_RATIO_CANONICAL,
        )
    } else {
        (
            PRUNE_LIST_MIN_ELEMENTS_COMPRESSED,
            PRUNE_LIST_MIN_SAVINGS_RATIO_COMPRESSED,
        )
    };
    if elements_len < min_elements {
        return Ok(None);
    }
    let len = array.len();
    if len == 0 {
        return Ok(None);
    }

    let first_offset = array.offset_at(0)?;
    let last_offset = array.offset_at(len)?;
    debug_assert!(first_offset <= last_offset);
    debug_assert!(last_offset <= elements_len);

    let referenced = last_offset - first_offset;
    let savings_ratio = 1.0 - (referenced as f64) / (elements_len as f64);
    if savings_ratio < min_savings {
        return Ok(None);
    }

    // Slice the elements to the live window.
    let new_elements = array.elements().slice(first_offset..last_offset)?;

    // Subtract `first_offset` from every offset, preserving the offsets dtype.
    let new_offsets: ArrayRef = if first_offset == 0 {
        array.offsets().clone()
    } else {
        match_each_integer_ptype!(array.offsets().dtype().as_ptype(), |O| {
            let scalar_value = <O as num_traits::FromPrimitive>::from_usize(first_offset)
                .ok_or_else(|| {
                    vortex_error::vortex_err!(
                        "first_offset {first_offset} does not fit in offsets type"
                    )
                })?;
            let constant = ConstantArray::new(
                Scalar::primitive(scalar_value, Nullability::NonNullable),
                len + 1,
            )
            .into_array();
            array.offsets().clone().binary(constant, Operator::Sub)?
        })
    };

    // SAFETY: We sliced the elements to exactly the live window and shifted offsets by the
    // same amount, so the (offset, length) invariants of `ListArray` are preserved.
    Ok(Some(unsafe {
        ListArray::new_unchecked(new_elements, new_offsets, array.list_validity())
    }))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn returns_none_for_small_elements() -> VortexResult<()> {
        let list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 5].into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_trim_unreferenced_elements(&list, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn returns_none_when_dense() -> VortexResult<()> {
        let elements = PrimitiveArray::from_iter(0i64..200_000).into_array();
        // offsets cover [0, 200000) — fully referenced.
        let offsets: Vec<u32> = (0..=2048).map(|i| (i * 97) as u32).collect();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_trim_unreferenced_elements(&list, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn trims_leading_and_trailing_garbage() -> VortexResult<()> {
        // 10_000_000 elements, but only [3_000_000, 3_100_000) is referenced (~99% unreferenced).
        let elements = PrimitiveArray::from_iter(0i64..10_000_000).into_array();
        let offsets: Vec<u32> = (0..=2000).map(|i| 3_000_000 + (i * 50) as u32).collect();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let trimmed = maybe_trim_unreferenced_elements(&list, &mut ctx)?
            .expect("expected a trim with leading+trailing garbage");
        assert_eq!(trimmed.elements().len(), 100_000);
        // The first list should be [3_000_000, ..., 3_000_049].
        let first = trimmed.list_elements_at(0)?;
        let first_canon = first.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(first_canon.len(), 50);
        assert_eq!(first_canon.as_slice::<i64>()[0], 3_000_000);
        Ok(())
    }
}
