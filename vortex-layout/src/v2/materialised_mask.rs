// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`MaterialisedMask`] + [`MaskRegistry`] — value-based distribution
//! of filter masks to pushdown consumers.
//!
//! ## Why a separate path from [`crate::v2::tee_stream::TeeStream`]
//!
//! Filter masks are *values*, not streams: every consumer wants the
//! whole mask for its row range; the source is small (1 bit/row); and
//! the canonical form is shareable. Treating it as a stream forced
//! per-consumer kanal channels, per-consumer producer fan-out work,
//! and per-consumer canonicalisation. For wide pushed-down scans
//! (Clickbench Q22 hits ~20 K consumers per scan), that overhead
//! dominates the actual filter compute.
//!
//! This module gives masks a value-based path:
//!
//! 1. The mask source is materialised **once per partition** into a
//!    canonical [`Mask`].
//! 2. The materialisation future is shared via [`futures::future::Shared`]
//!    — the first consumer to poll triggers it; subsequent consumers
//!    await the same future (cheap once it resolves).
//! 3. Consumers slice the canonical [`Mask`] in O(1) (Mask::slice is
//!    cheap on the underlying [`vortex_buffer::BitBuffer`]).
//!
//! [`crate::v2::let_use::LetPlan`] dispatches based on its source's
//! schema: `Bool` → `MaskRegistry`, anything else → `TeeStream`.
//! `TeeStream` stays in place for the (still-hypothetical) future
//! use case of sharing column streams.

use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::v2::let_use::LetId;
use crate::v2::scan_ctx::ScanCtxValue;

/// One partition's filter mask, materialised in canonical form.
/// Slicing is O(1) on the underlying bit buffer.
#[derive(Debug)]
pub struct MaterialisedMask {
    mask: Mask,
    row_range: Range<u64>,
}

impl MaterialisedMask {
    /// Build from a raw `ArrayRef` by canonicalising via the
    /// session's executor.
    pub fn from_array(
        array: ArrayRef,
        session: &VortexSession,
        row_range: Range<u64>,
    ) -> VortexResult<Self> {
        let mut ctx = session.create_execution_ctx();
        let mask: Mask = array.execute::<Mask>(&mut ctx)?;
        let expected_len = usize::try_from(row_range.end - row_range.start)?;
        if mask.len() != expected_len {
            vortex_bail!(
                "MaterialisedMask length {} does not match row range {row_range:?}",
                mask.len()
            );
        }
        Ok(Self { mask, row_range })
    }

    /// Slice the materialised mask to a sub-range. Inputs are in
    /// `u64` to match plan-level row coordinates.
    pub fn slice(&self, range: Range<u64>) -> VortexResult<Mask> {
        if range.start < self.row_range.start || range.end > self.row_range.end {
            vortex_bail!(
                "MaterialisedMask::slice range {range:?} is outside materialised range {:?}",
                self.row_range
            );
        }
        let start = usize::try_from(range.start - self.row_range.start)?;
        let end = usize::try_from(range.end - self.row_range.start)?;
        Ok(self.mask.slice(start..end))
    }

    /// Total row count covered by this mask.
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// True iff the mask covers zero rows.
    pub fn is_empty(&self) -> bool {
        self.mask.len() == 0
    }

    /// Borrow the underlying canonical [`Mask`].
    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    /// Absolute row range covered by this mask.
    pub fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }
}

/// A future that resolves to the materialised mask once. Multiple
/// consumers can await the same future — only the first poll runs
/// the source draining; subsequent polls hit the cached result.
type SharedMaskFuture = Shared<BoxFuture<'static, Result<Arc<MaterialisedMask>, Arc<VortexError>>>>;

/// Per-scan registry of mask materialisation futures, keyed by the
/// [`LetId`] under which each mask source was published.
#[derive(Default)]
pub struct MaskRegistry {
    cells: HashMap<LetId, MaskCell>,
}

struct MaskCell {
    row_range: Range<u64>,
    future: SharedMaskFuture,
}

impl Debug for MaskRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaskRegistry")
            .field("ids", &self.cells.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ScanCtxValue for MaskRegistry {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl MaskRegistry {
    /// Look up the shared materialisation future for `id`, if the
    /// registered materialisation covers `row_range`.
    pub fn get(&self, id: LetId, row_range: &Range<u64>) -> Option<SharedMaskFuture> {
        self.cells
            .get(&id)
            .filter(|cell| contains_range(&cell.row_range, row_range))
            .map(|cell| cell.future.clone())
    }

    /// Register a mask source under `id` for `row_range`. If an
    /// existing materialisation covers the requested range it is
    /// reused; otherwise it is replaced with a range-specific future.
    pub fn get_or_init(
        &mut self,
        id: LetId,
        row_range: Range<u64>,
        init: impl FnOnce() -> SharedMaskFuture,
    ) -> SharedMaskFuture {
        if let Some(cell) = self
            .cells
            .get(&id)
            .filter(|cell| contains_range(&cell.row_range, &row_range))
        {
            return cell.future.clone();
        }

        let future = init();
        self.cells.insert(
            id,
            MaskCell {
                row_range,
                future: future.clone(),
            },
        );
        future
    }
}

fn contains_range(outer: &Range<u64>, inner: &Range<u64>) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// Build a shared materialisation future from a `LayoutPlan` source
/// stream. The future drains the stream, canonicalises to a `Mask`,
/// and yields an `Arc<MaterialisedMask>`. Errors are also Arc-shared
/// so all awaiters see the same diagnostic.
pub fn build_materialise_future(
    source_stream: SendableArrayStream,
    session: VortexSession,
    row_range: Range<u64>,
) -> SharedMaskFuture {
    use vortex_array::stream::ArrayStreamExt as _;
    async move {
        let array = source_stream.read_all().await.map_err(Arc::new)?;
        let materialised =
            MaterialisedMask::from_array(array, &session, row_range).map_err(Arc::new)?;
        Ok(Arc::new(materialised))
    }
    .boxed()
    .shared()
}

/// Wrap an `Arc<MaterialisedMask>` slice as an `ArrayRef` so that
/// existing stream-based consumers (`FilteredFlatPlan`) can keep
/// using their `mask_stream.read_all()` path. The resulting
/// `BoolArray` shares the underlying `BitBuffer` with the
/// materialised mask — no copy.
pub fn slice_to_array(
    materialised: &MaterialisedMask,
    row_range: Range<u64>,
) -> VortexResult<ArrayRef> {
    let sliced = materialised.slice(row_range)?;
    Ok(BoolArray::new(sliced.to_bit_buffer(), Validity::NonNullable).into_array())
}
