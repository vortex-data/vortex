// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A patch set addressed by chunk-local indices.
//!
//! [`PatchesV2`] stores the same information as [`Patches`]: sparse exception values for an
//! array. It differs in how patch positions are addressed:
//!
//! - `indices` holds `u16` positions **local to each 1024-value chunk** instead of global row
//!   indices, so the index child stays two bytes per patch at any array length.
//! - `chunk_offsets` is required, holds `u32` prefix patch counts with a leading zero, and is
//!   rebased on every slice, so chunk lookups never need the saturating-adjustment bookkeeping
//!   that global offsets force onto [`Patches`].
//!
//! An `offset` in `0..PATCH_CHUNK_SIZE` places logical element zero inside the first chunk, so
//! slices at unaligned positions keep constant-time chunk addressing: logical index `i` lives at
//! grid position `offset + i`, in chunk `(offset + i) / 1024` at local position
//! `(offset + i) % 1024`.
//!
//! [`Patches`]: crate::patches::Patches

use std::ops::Range;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
#[cfg(any(test, feature = "_test-harness"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::validity::Validity;

static PATCHES_V2_SCATTER: AtomicBool = AtomicBool::new(false);

/// Returns whether decompression scatters chunked patches through [`PatchesV2`].
///
/// Converting a global-index patch set per decompression costs a pass and allocations over the
/// patch set, so the chunk-local scatter stays opt-in until the stored layout is chunk-local.
/// Enabled by [`force_patches_v2_scatter`] or `VORTEX_PATCHES_V2_SCATTER=1`.
pub fn use_patches_v2_scatter() -> bool {
    static FROM_ENV: LazyLock<bool> =
        LazyLock::new(|| std::env::var("VORTEX_PATCHES_V2_SCATTER").is_ok_and(|v| v == "1"));
    PATCHES_V2_SCATTER.load(Ordering::Relaxed) || *FROM_ENV
}

/// Force the chunk-local patch scatter on or off for this process.
pub fn force_patches_v2_scatter(enabled: bool) {
    PATCHES_V2_SCATTER.store(enabled, Ordering::Relaxed);
}

#[cfg(any(test, feature = "_test-harness"))]
static SCATTERS: AtomicU64 = AtomicU64::new(0);

/// The number of [`PatchesV2::apply_into`] scatters performed, for tests that need to prove a
/// read path took the chunk-local form rather than silently falling back.
#[cfg(any(test, feature = "_test-harness"))]
pub fn patches_v2_scatter_count() -> u64 {
    SCATTERS.load(Ordering::Relaxed)
}

/// Sparse patch values addressed by chunk-local `u16` indices.
#[derive(Debug, Clone)]
pub struct PatchesV2 {
    array_len: usize,
    /// Grid position of logical element zero, in `0..PATCH_CHUNK_SIZE`.
    offset: usize,
    /// Chunk-local `u16` patch positions, sorted within each chunk.
    indices: ArrayRef,
    /// One patch value per index.
    values: ArrayRef,
    /// `u32` prefix patch counts per chunk, with a leading zero.
    chunk_offsets: ArrayRef,
}

impl PatchesV2 {
    /// Construct and validate a new patch set.
    ///
    /// Validation canonicalizes the index and chunk-offset children, so callers on a hot path
    /// with already-validated components should prefer [`Self::new_unchecked`].
    pub fn try_new(
        array_len: usize,
        offset: usize,
        indices: ArrayRef,
        values: ArrayRef,
        chunk_offsets: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offset < PATCH_CHUNK_SIZE,
            "PatchesV2 offset must be within the first chunk"
        );
        vortex_ensure!(
            indices.len() == values.len(),
            "PatchesV2 indices and values must have the same length"
        );
        vortex_ensure!(!indices.is_empty(), "PatchesV2 must not be empty");
        vortex_ensure!(
            indices.len() <= array_len,
            "PatchesV2 cannot have more patches than rows"
        );
        vortex_ensure!(
            indices.dtype() == &DType::Primitive(PType::U16, NonNullable),
            "PatchesV2 indices must be non-nullable u16, got {}",
            indices.dtype()
        );
        vortex_ensure!(
            chunk_offsets.dtype() == &DType::Primitive(PType::U32, NonNullable),
            "PatchesV2 chunk offsets must be non-nullable u32, got {}",
            chunk_offsets.dtype()
        );
        let chunk_count = (offset + array_len).div_ceil(PATCH_CHUNK_SIZE);
        vortex_ensure!(
            chunk_offsets.len() == chunk_count + 1,
            "PatchesV2 expects {} chunk offsets, got {}",
            chunk_count + 1,
            chunk_offsets.len()
        );

        let local_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let local_indices = local_indices.as_slice::<u16>();
        let offsets = chunk_offsets.clone().execute::<PrimitiveArray>(ctx)?;
        let offsets = offsets.as_slice::<u32>();
        vortex_ensure!(
            offsets.first() == Some(&0),
            "PatchesV2 chunk offsets must start at zero"
        );
        vortex_ensure!(
            usize::try_from(offsets[chunk_count])? == indices.len(),
            "PatchesV2 chunk offsets must end at the patch count"
        );
        for chunk_idx in 0..chunk_count {
            let chunk =
                usize::try_from(offsets[chunk_idx])?..usize::try_from(offsets[chunk_idx + 1])?;
            vortex_ensure!(
                chunk.start <= chunk.end,
                "PatchesV2 chunk offsets must not decrease"
            );
            let chunk_grid_len = grid_range(offset, array_len, chunk_idx, chunk_count);
            let locals = &local_indices[chunk];
            vortex_ensure!(
                locals.windows(2).all(|pair| pair[0] < pair[1]),
                "PatchesV2 indices must be strictly sorted within each chunk"
            );
            vortex_ensure!(
                locals
                    .iter()
                    .all(|&local| chunk_grid_len.contains(&usize::from(local))),
                "PatchesV2 chunk {chunk_idx} contains out-of-range indices"
            );
        }

        Ok(unsafe { Self::new_unchecked(array_len, offset, indices, values, chunk_offsets) })
    }

    /// Construct a patch set without validating the components.
    ///
    /// # Safety
    ///
    /// Callers must uphold every invariant checked by [`Self::try_new`]: matching child lengths,
    /// non-nullable `u16` indices strictly sorted within each chunk and inside the sliced grid
    /// range, and non-decreasing `u32` chunk offsets starting at zero and ending at the patch
    /// count, with one entry per chunk plus one.
    pub unsafe fn new_unchecked(
        array_len: usize,
        offset: usize,
        indices: ArrayRef,
        values: ArrayRef,
        chunk_offsets: ArrayRef,
    ) -> Self {
        Self {
            array_len,
            offset,
            indices,
            values,
            chunk_offsets,
        }
    }

    /// Convert a global-index [`Patches`] into chunk-local form.
    pub fn from_patches(patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let array_len = patches.array_len();
        let offset = patches.offset() % PATCH_CHUNK_SIZE;
        let chunk_count = (offset + array_len).div_ceil(PATCH_CHUNK_SIZE);
        let global = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let mut locals = Vec::with_capacity(global.len());
        let mut chunk_offsets = vec![0u32; chunk_count + 1];
        let patches_offset = patches.offset();
        crate::match_each_unsigned_integer_ptype!(global.ptype(), |P| {
            for &index in global.as_slice::<P>() {
                // Rebase from the source offset onto this grid, which starts at `offset`.
                let index: usize = index.as_();
                let grid = index - patches_offset + offset;
                locals.push(u16::try_from(grid % PATCH_CHUNK_SIZE)?);
                chunk_offsets[grid / PATCH_CHUNK_SIZE + 1] += 1;
            }
        });
        for chunk_idx in 0..chunk_count {
            chunk_offsets[chunk_idx + 1] += chunk_offsets[chunk_idx];
        }
        Ok(unsafe {
            Self::new_unchecked(
                array_len,
                offset,
                PrimitiveArray::new(Buffer::from(locals), Validity::NonNullable).into_array(),
                patches.values().clone(),
                PrimitiveArray::new(Buffer::from(chunk_offsets), Validity::NonNullable)
                    .into_array(),
            )
        })
    }

    /// Convert back into a global-index [`Patches`].
    pub fn to_patches(&self, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
        let (locals, offsets) = self.canonical_parts(ctx)?;
        let mut globals = Vec::with_capacity(locals.len());
        for chunk_idx in 0..offsets.len() - 1 {
            let chunk =
                usize::try_from(offsets[chunk_idx])?..usize::try_from(offsets[chunk_idx + 1])?;
            for &local in &locals[chunk] {
                globals.push(u64::try_from(
                    chunk_idx * PATCH_CHUNK_SIZE + usize::from(local) - self.offset,
                )?);
            }
        }
        Patches::new(
            self.array_len,
            0,
            PrimitiveArray::new(Buffer::from(globals), Validity::NonNullable).into_array(),
            self.values.clone(),
            None,
        )
    }

    /// Returns the length of the patched array.
    pub fn array_len(&self) -> usize {
        self.array_len
    }

    /// Returns the number of patches.
    pub fn num_patches(&self) -> usize {
        self.indices.len()
    }

    /// Returns the dtype of the patch values.
    pub fn dtype(&self) -> &DType {
        self.values.dtype()
    }

    /// Returns the chunk-local patch indices.
    pub fn indices(&self) -> &ArrayRef {
        &self.indices
    }

    /// Returns the patch values.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// Returns the per-chunk patch count prefix sums.
    pub fn chunk_offsets(&self) -> &ArrayRef {
        &self.chunk_offsets
    }

    /// Returns the grid position of logical element zero.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Search for a patch at logical `index`.
    ///
    /// Returns [`SearchResult::Found`] with the patch ordinal, or [`SearchResult::NotFound`]
    /// with the insertion point.
    pub fn search_index(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<SearchResult> {
        if let Some(view) = self.view() {
            return Ok(view.search_index(index));
        }
        if index >= self.array_len {
            return Ok(SearchResult::NotFound(self.num_patches()));
        }
        let (locals, offsets) = self.canonical_parts(ctx)?;
        Ok(search_local(&locals, &offsets, self.offset + index))
    }

    /// Borrow a resolved view over canonical index and chunk-offset children.
    ///
    /// Returns `None` when either child is not a canonical primitive array. Hot loops should
    /// resolve the view once and query it repeatedly; each call performs the downcasts.
    pub fn view(&self) -> Option<PatchesV2View<'_>> {
        let locals = self.indices.as_opt::<Primitive>()?;
        let offsets = self.chunk_offsets.as_opt::<Primitive>()?;
        Some(PatchesV2View {
            locals,
            offsets,
            offset: self.offset,
            array_len: self.array_len,
        })
    }

    /// Visit every patch as `(logical_index, patch_ordinal)`, in patch order.
    ///
    /// This is the decompression primitive: callers scatter the canonicalized patch values over
    /// a decoded buffer without materializing global indices.
    pub fn apply_each(
        &self,
        ctx: &mut ExecutionCtx,
        mut apply: impl FnMut(usize, usize),
    ) -> VortexResult<()> {
        if let Some(view) = self.view() {
            apply_each_parts(
                view.locals.as_slice::<u16>(),
                view.offsets.as_slice::<u32>(),
                self.offset,
                &mut apply,
            );
            return Ok(());
        }
        let (locals, offsets) = self.canonical_parts(ctx)?;
        apply_each_parts(&locals, &offsets, self.offset, &mut apply);
        Ok(())
    }

    /// Scatter the patch values over `out`, which holds the decoded base values.
    ///
    /// This is the decompression scatter for canonical primitive output: each patched position
    /// is overwritten with its patch value.
    pub fn apply_into<T: crate::dtype::NativePType>(
        &self,
        out: &mut [T],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            out.len() == self.array_len,
            "PatchesV2 apply_into expects {} elements, got {}",
            self.array_len,
            out.len()
        );
        #[cfg(any(test, feature = "_test-harness"))]
        SCATTERS.fetch_add(1, Ordering::Relaxed);
        let values = self.values.clone().execute::<PrimitiveArray>(ctx)?;
        let values = values.as_slice::<T>();
        self.apply_each(ctx, |logical, ordinal| out[logical] = values[ordinal])
    }

    /// Return the patch value at logical `index`, if one exists.
    pub fn get_patched(
        &self,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        self.search_index(index, ctx)?
            .to_found()
            .map(|patch_idx| self.values.execute_scalar(patch_idx, ctx))
            .transpose()
    }

    /// Slice the patch set to `range`, returning `None` when no patches remain.
    ///
    /// The chunk offsets are rebased so the result is self-contained: no saturating adjustments
    /// are carried forward, unlike [`Patches::slice`].
    pub fn slice(&self, range: Range<usize>, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        vortex_ensure!(
            range.end <= self.array_len,
            "PatchesV2 slice is out of bounds"
        );
        if range.is_empty() {
            return Ok(None);
        }
        let (locals, offsets) = self.canonical_parts(ctx)?;
        let grid_start = self.offset + range.start;
        let grid_end = self.offset + range.end;
        let patch_start = search_local(&locals, &offsets, grid_start).to_index();
        let patch_end = search_local(&locals, &offsets, grid_end).to_index();
        if patch_start == patch_end {
            return Ok(None);
        }

        let chunk_start = grid_start / PATCH_CHUNK_SIZE;
        let chunk_end = grid_end.div_ceil(PATCH_CHUNK_SIZE);
        let rebased: Vec<u32> = (chunk_start..=chunk_end)
            .map(|chunk_idx| {
                let offset = usize::try_from(offsets[chunk_idx])?.clamp(patch_start, patch_end)
                    - patch_start;
                Ok(u32::try_from(offset)?)
            })
            .collect::<VortexResult<_>>()?;
        Ok(Some(unsafe {
            Self::new_unchecked(
                range.len(),
                grid_start % PATCH_CHUNK_SIZE,
                self.indices.slice(patch_start..patch_end)?,
                self.values.slice(patch_start..patch_end)?,
                PrimitiveArray::new(Buffer::from(rebased), Validity::NonNullable).into_array(),
            )
        }))
    }

    /// Execute the index and chunk-offset children into typed buffers.
    ///
    /// This is the slow path for encoded children; canonical children are read in place by the
    /// callers' downcast fast paths.
    fn canonical_parts(&self, ctx: &mut ExecutionCtx) -> VortexResult<(Buffer<u16>, Buffer<u32>)> {
        let locals = self
            .indices
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_buffer::<u16>();
        let offsets = self
            .chunk_offsets
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_buffer::<u32>();
        Ok((locals, offsets))
    }
}

/// A resolved, borrowed view over a [`PatchesV2`] with canonical children.
///
/// Constructed via [`PatchesV2::view`]; queries are plain slice reads with no dispatch,
/// allocation, or error paths, so this is the form hot loops should hold.
#[derive(Clone, Debug)]
pub struct PatchesV2View<'a> {
    locals: ArrayView<'a, Primitive>,
    offsets: ArrayView<'a, Primitive>,
    offset: usize,
    array_len: usize,
}

impl PatchesV2View<'_> {
    /// Search for a patch at logical `index`.
    pub fn search_index(&self, index: usize) -> SearchResult {
        if index >= self.array_len {
            return SearchResult::NotFound(self.locals.len());
        }
        search_local(
            self.locals.as_slice::<u16>(),
            self.offsets.as_slice::<u32>(),
            self.offset + index,
        )
    }

    /// Returns the patch ordinal at logical `index`, if one exists.
    pub fn patch_ordinal(&self, index: usize) -> Option<usize> {
        self.search_index(index).to_found()
    }
}

/// Walk every patch as `(logical_index, patch_ordinal)` from resolved parts.
fn apply_each_parts(
    locals: &[u16],
    offsets: &[u32],
    offset: usize,
    apply: &mut impl FnMut(usize, usize),
) {
    // Walk patches with a chunk cursor so sparse patch sets skip empty chunks cheaply.
    let mut chunk_idx = 0usize;
    for (ordinal, &local) in locals.iter().enumerate() {
        while offsets[chunk_idx + 1] as usize <= ordinal {
            chunk_idx += 1;
        }
        apply(
            chunk_idx * PATCH_CHUNK_SIZE + usize::from(local) - offset,
            ordinal,
        );
    }
}

/// The grid-local index range a chunk may address, honoring first- and last-chunk trims.
fn grid_range(
    offset: usize,
    array_len: usize,
    chunk_idx: usize,
    chunk_count: usize,
) -> Range<usize> {
    let start = if chunk_idx == 0 { offset } else { 0 };
    let stop = if chunk_idx == chunk_count - 1 {
        (offset + array_len) - chunk_idx * PATCH_CHUNK_SIZE
    } else {
        PATCH_CHUNK_SIZE
    };
    start..stop
}

/// Search the flat local-index buffer for grid position `grid`.
///
/// Chunk selection is constant time via the offsets; the in-chunk search is a binary search
/// over at most [`PATCH_CHUNK_SIZE`] `u16` values.
fn search_local(locals: &[u16], offsets: &[u32], grid: usize) -> SearchResult {
    let chunk_idx = grid / PATCH_CHUNK_SIZE;
    if chunk_idx >= offsets.len() - 1 {
        return SearchResult::NotFound(locals.len());
    }
    let chunk = offsets[chunk_idx] as usize..offsets[chunk_idx + 1] as usize;
    let local =
        u16::try_from(grid % PATCH_CHUNK_SIZE).vortex_expect("chunk-local index fits in u16");
    match locals[chunk.clone()].binary_search(&local) {
        Ok(idx) => SearchResult::Found(chunk.start + idx),
        Err(idx) => SearchResult::NotFound(chunk.start + idx),
    }
}

#[cfg(test)]
mod tests;
