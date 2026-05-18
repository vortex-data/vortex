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
//!
//! ### Cost model
//!
//! The decision of whether to commit to a trim depends on the downstream export cost, which
//! varies by destination and dtype. The caller supplies a [`TrimCostModel`] to indicate which
//! regime applies:
//!
//! - **Zero-copy canonical destinations** (Arrow): a canonical primitive / varbinview /
//!   decimal / fixed-size-list buffer is passed to Arrow by reference. The trim saves no CPU
//!   for canonical children — only memory. We fire only on overwhelming waste.
//! - **Per-element-paying canonical destinations** (DuckDB primitive copy, struct field
//!   copies): each unreferenced element saves `dtype.element_size()` bytes of memcpy work, so
//!   wide dtypes justify firing at lower savings ratios than narrow ones.
//! - **Compressed elements** (any destination): per-position decompression dominates the
//!   export cost. We fire at a moderate (50%) savings ratio regardless of caller.

use vortex_error::VortexResult;

use super::ListArray;
use super::ListArrayExt;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Caller-supplied cost estimate for downstream export work, used by
/// [`maybe_trim_unreferenced_elements`] to decide when the trim's fixed cost pays back.
#[derive(Clone, Copy, Debug)]
pub struct TrimCostModel {
    /// Estimated bytes the downstream export does *per element* for canonical children.
    /// Pass `0` for zero-copy destinations (Arrow → Arrow); pass `dtype.element_size()` for
    /// per-element-copy destinations (DuckDB primitive memcpy, struct field copies, …).
    ///
    /// For compressed children this hint is ignored — per-position decompression dominates
    /// and uses a separate fixed-ratio threshold.
    pub canonical_bytes_per_element: usize,
}

impl TrimCostModel {
    /// Cost model for Arrow → Arrow exports: canonical children are passed zero-copy, so the
    /// trim only fires on overwhelming waste (memory-driven, not CPU-driven).
    pub const fn arrow() -> Self {
        Self {
            canonical_bytes_per_element: 0,
        }
    }

    /// Cost model for DuckDB exports.
    ///
    /// - **Fixed-width canonical children** (`Primitive`, `Decimal`, `FixedSizeList<…>`,
    ///   struct of fixed-width fields): DuckDB exports zero-copy by registering the existing
    ///   buffer and setting a pointer (see `PrimitiveExporter::export`). Per-element work is
    ///   ~zero, so we mark this `0` and the trim only fires on overwhelming waste — same as
    ///   Arrow.
    /// - **Variable-width canonical children** (`Utf8`, `Binary`, `List`, `Variant`): DuckDB
    ///   does a per-view / per-list-entry struct copy of ~16 bytes into the vector plus
    ///   buffer registration. Per-element cost matters, so a sparse prefix/suffix dominates.
    ///   We use the 16-byte per-view metadata as the per-unreferenced-element saving
    ///   estimate.
    pub fn for_dtype(elements_dtype: &DType) -> Self {
        const VARWIDTH_PER_ELEMENT_BYTES: usize = 16;
        let canonical_bytes_per_element = if elements_dtype.element_size().is_some() {
            0
        } else {
            VARWIDTH_PER_ELEMENT_BYTES
        };
        Self {
            canonical_bytes_per_element,
        }
    }
}

/// Minimum elements buffer length before we even consider trimming, regardless of cost model.
///
/// Below this the trim's allocator / arithmetic overhead dominates any plausible savings.
pub const PRUNE_LIST_MIN_ELEMENTS: usize = 4 * 1024;

/// Savings ratio threshold for the **zero-copy canonical** path (`canonical_bytes_per_element
/// = 0`). The trim then saves only memory, not CPU, so we require an overwhelming fraction of
/// the buffer to be unreferenced before paying the rewrite cost.
pub const PRUNE_LIST_OVERWHELMING_SAVINGS_RATIO: f64 = 0.97;

/// Minimum elements buffer length for the zero-copy canonical path, alongside the
/// overwhelming-savings ratio. Calibrated to amortise the trim's per-call fixed cost on
/// realistic Vortex chunk sizes.
pub const PRUNE_LIST_OVERWHELMING_MIN_ELEMENTS: usize = 64 * 1024;

/// Bytes-saved threshold for the **per-element-paying canonical** path
/// (`canonical_bytes_per_element > 0`). Each unreferenced element contributes
/// `canonical_bytes_per_element` to the estimate; once the total exceeds this, the trim's
/// downstream savings reliably exceed its `O(num_lists)` offsets-rewrite cost.
pub const PRUNE_LIST_BYTES_SAVED_TARGET: usize = 32 * 1024;

/// Savings ratio threshold for the **compressed-elements** path. Per-position decompression
/// dominates the export cost, so even moderate waste justifies the trim. The trim's
/// child-encoding-aware `take` keeps compressed data compressed for the discarded positions.
pub const PRUNE_LIST_COMPRESSED_SAVINGS_RATIO: f64 = 0.50;

/// Minimum elements buffer length for the compressed-elements path. Below this the rewrite's
/// fixed cost is comparable to the saved decompression.
pub const PRUNE_LIST_COMPRESSED_MIN_ELEMENTS: usize = 32 * 1024;

/// Inspect `array` and, if the prefix/suffix outside `[offsets[0], offsets[len])` represents
/// enough discarded work to amortise the trim's fixed cost, return a trimmed [`ListArray`]
/// whose elements contain only the referenced positions.
///
/// `cost` is the caller's downstream cost model — see [`TrimCostModel`] for the cases. The
/// canonical / compressed split is detected internally via `elements.is_canonical()`.
pub fn maybe_trim_unreferenced_elements(
    array: &ListArray,
    cost: TrimCostModel,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ListArray>> {
    let elements_len = array.elements().len();
    if elements_len < PRUNE_LIST_MIN_ELEMENTS {
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
    let unreferenced = elements_len - referenced;

    let canonical_elements = array.elements().is_canonical();
    let commit = if canonical_elements {
        if cost.canonical_bytes_per_element == 0 {
            // Zero-copy export: only fire on overwhelming waste (memory-only payoff).
            let savings_ratio = (unreferenced as f64) / (elements_len as f64);
            elements_len >= PRUNE_LIST_OVERWHELMING_MIN_ELEMENTS
                && savings_ratio >= PRUNE_LIST_OVERWHELMING_SAVINGS_RATIO
        } else {
            // Per-element-paying export: bytes-saved scales with dtype width.
            let bytes_saved = unreferenced.saturating_mul(cost.canonical_bytes_per_element);
            bytes_saved >= PRUNE_LIST_BYTES_SAVED_TARGET
        }
    } else {
        // Compressed elements: per-position decompression dominates regardless of cost model.
        let savings_ratio = (unreferenced as f64) / (elements_len as f64);
        elements_len >= PRUNE_LIST_COMPRESSED_MIN_ELEMENTS
            && savings_ratio >= PRUNE_LIST_COMPRESSED_SAVINGS_RATIO
    };
    if !commit {
        return Ok(None);
    }

    // Slice the elements to the live window.
    let new_elements = array.elements().slice(first_offset..last_offset)?;

    // Subtract `first_offset` from every offset, preserving the offsets dtype. The general
    // `binary(offsets, ConstantArray, Sub)` path goes through the full compute framework
    // (scalar-fn wrapper → optimize → kernel dispatch) which costs ~µs of fixed overhead per
    // call. For the common case where offsets is already a host `Primitive`, we do the
    // subtract inline with `try_into_buffer_mut` (zero-copy when uniquely owned, single copy
    // otherwise) — orders of magnitude faster than the framework path.
    let new_offsets: ArrayRef = if first_offset == 0 {
        array.offsets().clone()
    } else if let Some(prim) = array.offsets().as_opt::<crate::arrays::Primitive>()
        && array.offsets().is_host()
    {
        let prim = prim.into_owned();
        let validity = prim.validity()?;
        match_each_integer_ptype!(prim.ptype(), |O| {
            let shift: O =
                num_traits::FromPrimitive::from_usize(first_offset).ok_or_else(|| {
                    vortex_error::vortex_err!(
                        "first_offset {first_offset} does not fit in offsets type"
                    )
                })?;
            let mut buf = prim.into_buffer_mut::<O>();
            for o in buf.iter_mut() {
                *o -= shift;
            }
            crate::arrays::PrimitiveArray::new(buf.freeze(), validity).into_array()
        })
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

    fn arrow_cost() -> TrimCostModel {
        TrimCostModel::arrow()
    }

    fn duckdb_cost_i64() -> TrimCostModel {
        // Fixed-width → zero-copy DuckDB export → cost 0 → same threshold as Arrow.
        TrimCostModel::for_dtype(&DType::Primitive(
            crate::dtype::PType::I64,
            Nullability::NonNullable,
        ))
    }

    fn duckdb_cost_varbin() -> TrimCostModel {
        // Variable-width → per-view DuckDB export cost → bytes-saved threshold applies.
        TrimCostModel::for_dtype(&DType::Utf8(Nullability::NonNullable))
    }

    #[test]
    fn returns_none_for_small_elements() -> VortexResult<()> {
        let list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 5].into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_trim_unreferenced_elements(&list, arrow_cost(), &mut ctx)?.is_none());
        assert!(maybe_trim_unreferenced_elements(&list, duckdb_cost_i64(), &mut ctx)?.is_none());
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
        assert!(maybe_trim_unreferenced_elements(&list, arrow_cost(), &mut ctx)?.is_none());
        assert!(maybe_trim_unreferenced_elements(&list, duckdb_cost_i64(), &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn trims_leading_and_trailing_garbage_for_either_cost_model() -> VortexResult<()> {
        // 10_000_000 elements, but only [3_000_000, 3_100_000) is referenced (~99% unreferenced).
        let elements = PrimitiveArray::from_iter(0i64..10_000_000).into_array();
        let offsets: Vec<u32> = (0..=2000).map(|i| 3_000_000 + (i * 50) as u32).collect();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let trimmed = maybe_trim_unreferenced_elements(&list, arrow_cost(), &mut ctx)?
            .expect("Arrow cost model trims on overwhelming savings");
        assert_eq!(trimmed.elements().len(), 100_000);
        let trimmed = maybe_trim_unreferenced_elements(&list, duckdb_cost_i64(), &mut ctx)?
            .expect("DuckDB cost model trims on overwhelming savings too");
        assert_eq!(trimmed.elements().len(), 100_000);
        // The first list should be [3_000_000, ..., 3_000_049].
        let first = trimmed.list_elements_at(0)?;
        let first_canon = first.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(first_canon.len(), 50);
        assert_eq!(first_canon.as_slice::<i64>()[0], 3_000_000);
        Ok(())
    }

    /// Wide-canonical DuckDB-style case: 50% savings on a 128 KiB `i64` buffer. Both Arrow
    /// and DuckDB skip — primitive is zero-copy in *both* destinations.
    #[test]
    fn fixed_width_canonical_skipped_below_overwhelming() -> VortexResult<()> {
        // 16 384 elements × 8 bytes = 128 KiB; lists cover the middle 50% = 8 192 elements.
        let elements = PrimitiveArray::from_iter(0i64..16_384).into_array();
        let offsets: Vec<u32> = (4096..=12_288).step_by(8).map(|x| x as u32).collect();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_trim_unreferenced_elements(&list, arrow_cost(), &mut ctx)?.is_none());
        assert!(maybe_trim_unreferenced_elements(&list, duckdb_cost_i64(), &mut ctx)?.is_none());
        Ok(())
    }

    /// Variable-width canonical case: 50% savings on a 128 KiB-positions varbin buffer.
    /// Arrow stays zero-copy (skip); DuckDB does per-view metadata copies — `8192 × 16 = 128
    /// KiB` of saved per-view work clears the 32 KiB threshold → commit.
    #[test]
    fn duckdb_cost_trims_variable_width_canonical_at_50pct_savings() -> VortexResult<()> {
        // Build a varbin-typed list with 16384 elements, lists cover middle 50%.
        // Use VarBin (or VarBinView) with placeholder data so element_size() returns None.
        use crate::arrays::VarBinViewArray;
        let strings: Vec<String> = (0..16_384).map(|i| format!("v{i}")).collect();
        let elements =
            VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
        let offsets: Vec<u32> = (4096..=12_288).step_by(8).map(|x| x as u32).collect();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_trim_unreferenced_elements(&list, arrow_cost(), &mut ctx)?.is_none());
        let trimmed = maybe_trim_unreferenced_elements(&list, duckdb_cost_varbin(), &mut ctx)?
            .expect("DuckDB varbin cost model trims when bytes saved clears threshold");
        assert_eq!(trimmed.elements().len(), 8192);
        Ok(())
    }
}
