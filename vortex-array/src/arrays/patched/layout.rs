// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk-local patch addressing shared by every [`Patched`](super::Patched) code path.
//!
//! Logical row `i` of a patched array lives at grid position `offset + i`, in chunk
//! `(offset + i) / PATCH_CHUNK_SIZE` at local index `(offset + i) % PATCH_CHUNK_SIZE`. Chunk `c`
//! owns the patch ordinals `chunk_offsets[c]..chunk_offsets[c + 1]`, which index directly into
//! the `patch_indices` and `patch_values` children.
//!
//! Slicing shares the patch children and slices only `chunk_offsets`, so the first and last chunk
//! may hold patches whose grid position falls outside `offset..offset + len`. Those patches are
//! *dead*: every reader skips them, and [`PatchedView::live`] reports the ordinal range that
//! survives.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::dtype::IntegerPType;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::search_sorted::SearchResult;

/// The number of chunks spanned by `len` rows starting at grid position `offset`.
pub(crate) fn n_chunks(offset: usize, len: usize) -> usize {
    (offset + len).div_ceil(PATCH_CHUNK_SIZE)
}

/// A borrowed view over the canonical position children of a patched array.
///
/// Every query is a plain slice read: constant-time chunk selection followed by a binary search
/// over at most [`PATCH_CHUNK_SIZE`] local indices. Hot loops should build one view and reuse it.
#[derive(Clone, Copy, Debug)]
pub struct PatchedView<'a> {
    offset: usize,
    len: usize,
    indices: &'a [u16],
    chunk_offsets: &'a [u32],
}

impl<'a> PatchedView<'a> {
    /// Build a view over already-validated position children.
    pub fn new(offset: usize, len: usize, indices: &'a [u16], chunk_offsets: &'a [u32]) -> Self {
        debug_assert!(offset < PATCH_CHUNK_SIZE);
        debug_assert_eq!(chunk_offsets.len(), n_chunks(offset, len) + 1);
        Self {
            offset,
            len,
            indices,
            chunk_offsets,
        }
    }

    /// Grid position of logical row zero.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of logical rows in view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view holds no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of chunks addressed by the view.
    pub fn n_chunks(&self) -> usize {
        self.chunk_offsets.len() - 1
    }

    /// Chunk-local patch indices, including any dead patches.
    pub fn indices(&self) -> &'a [u16] {
        self.indices
    }

    /// Prefix patch counts, one entry per chunk plus a terminator.
    pub fn chunk_offsets(&self) -> &'a [u32] {
        self.chunk_offsets
    }

    /// The patch ordinals owned by `chunk`.
    pub fn chunk(&self, chunk: usize) -> Range<usize> {
        self.chunk_offsets[chunk] as usize..self.chunk_offsets[chunk + 1] as usize
    }

    /// Search for a patch at logical `index`.
    ///
    /// Returns [`SearchResult::Found`] with the patch ordinal, or [`SearchResult::NotFound`] with
    /// the insertion point. Indices at or beyond the view report the end of the patch list.
    pub fn search(&self, index: usize) -> SearchResult {
        if index >= self.len {
            return SearchResult::NotFound(self.indices.len());
        }
        let grid = self.offset + index;
        let chunk = self.chunk(grid / PATCH_CHUNK_SIZE);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a position within a chunk always fits in u16"
        )]
        let local = (grid % PATCH_CHUNK_SIZE) as u16;
        match self.indices[chunk.clone()].binary_search(&local) {
            Ok(idx) => SearchResult::Found(chunk.start + idx),
            Err(idx) => SearchResult::NotFound(chunk.start + idx),
        }
    }

    /// The ordinal range of patches whose row falls inside the view.
    ///
    /// Dead patches can only form a prefix of the first chunk and a suffix of the last chunk,
    /// because indices are sorted within each chunk.
    pub fn live(&self) -> Range<usize> {
        if self.len == 0 {
            return 0..0;
        }
        let n_chunks = self.n_chunks();

        let first = self.chunk(0);
        let start = first.start
            + self.indices[first].partition_point(|&local| usize::from(local) < self.offset);

        let last = self.chunk(n_chunks - 1);
        let end_local = self.offset + self.len - (n_chunks - 1) * PATCH_CHUNK_SIZE;
        let end = last.start
            + self.indices[last].partition_point(|&local| usize::from(local) < end_local);

        start..end.max(start)
    }

    /// Visit every in-view patch as `(row, ordinal)` in row order.
    pub fn for_each(&self, mut visit: impl FnMut(usize, usize)) {
        for chunk_idx in 0..self.n_chunks() {
            let base = chunk_idx * PATCH_CHUNK_SIZE;
            for ordinal in self.chunk(chunk_idx) {
                let Some(row) =
                    (base + usize::from(self.indices[ordinal])).checked_sub(self.offset)
                else {
                    continue;
                };
                // Indices are sorted within the chunk, so the remaining patches are dead too.
                if row >= self.len {
                    break;
                }
                visit(row, ordinal);
            }
        }
    }
}

/// Check every invariant of the chunk-local layout against canonical position children.
pub(crate) fn validate_layout(
    offset: usize,
    len: usize,
    indices: &[u16],
    chunk_offsets: &[u32],
) -> VortexResult<()> {
    vortex_ensure!(
        offset < PATCH_CHUNK_SIZE,
        "Patched offset {offset} must be within the first chunk"
    );
    let n_chunks = n_chunks(offset, len);
    vortex_ensure!(
        chunk_offsets.len() == n_chunks + 1,
        "Patched expects {} chunk offsets, got {}",
        n_chunks + 1,
        chunk_offsets.len()
    );
    vortex_ensure!(
        chunk_offsets.windows(2).all(|pair| pair[0] <= pair[1]),
        "Patched chunk offsets must not decrease"
    );
    vortex_ensure!(
        chunk_offsets
            .last()
            .is_some_and(|&last| last as usize <= indices.len()),
        "Patched chunk offsets exceed the {} patches",
        indices.len()
    );
    for chunk_idx in 0..n_chunks {
        let chunk =
            &indices[chunk_offsets[chunk_idx] as usize..chunk_offsets[chunk_idx + 1] as usize];
        vortex_ensure!(
            chunk
                .iter()
                .all(|&local| usize::from(local) < PATCH_CHUNK_SIZE),
            "Patched chunk {chunk_idx} holds an index outside the chunk"
        );
        vortex_ensure!(
            chunk.windows(2).all(|pair| pair[0] < pair[1]),
            "Patched indices must be strictly increasing within chunk {chunk_idx}"
        );
    }
    Ok(())
}

/// Chunk-local position children built from sorted global patch indices.
pub(crate) struct ChunkLocal {
    /// Grid position of logical row zero.
    pub offset: usize,
    pub indices: Vec<u16>,
    pub chunk_offsets: Vec<u32>,
}

/// Convert sorted global indices addressed from `patches_offset` into chunk-local form.
///
/// The grid keeps the global indices' alignment: logical row zero lands at
/// `patches_offset % PATCH_CHUNK_SIZE`, so patch chunks line up with the chunks of an inner
/// encoding that was sliced by the same offset.
pub(crate) fn chunk_local_from_global<I: IntegerPType>(
    global: &[I],
    patches_offset: usize,
    array_len: usize,
) -> VortexResult<ChunkLocal> {
    let offset = patches_offset % PATCH_CHUNK_SIZE;
    let n_chunks = n_chunks(offset, array_len);
    let mut indices = Vec::with_capacity(global.len());
    let mut chunk_offsets = vec![0u32; n_chunks + 1];
    for &index in global {
        let index: usize = index.as_();
        let Some(row) = index
            .checked_sub(patches_offset)
            .filter(|&row| row < array_len)
        else {
            vortex_bail!(
                "patch index {index} is outside the array starting at {patches_offset} with {array_len} rows"
            );
        };
        let grid = offset + row;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a position within a chunk always fits in u16"
        )]
        indices.push((grid % PATCH_CHUNK_SIZE) as u16);
        chunk_offsets[grid / PATCH_CHUNK_SIZE + 1] += 1;
    }
    for chunk_idx in 0..n_chunks {
        chunk_offsets[chunk_idx + 1] += chunk_offsets[chunk_idx];
    }
    Ok(ChunkLocal {
        offset,
        indices,
        chunk_offsets,
    })
}
