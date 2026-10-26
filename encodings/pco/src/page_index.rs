// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::PcoMetadata;

#[derive(Debug)]
pub(crate) enum PageIndex {
    Legacy,
    IndexedFast(Arc<IndexedIndex>),
}

impl PageIndex {
    pub fn try_new(metadata: Arc<PcoMetadata>) -> VortexResult<Self> {
        if metadata.logical_row_prefix.is_none() {
            return Ok(Self::Legacy);
        }
        let mut chunk_ends = Vec::with_capacity(metadata.chunks.len());
        let mut end = 0usize;
        for chunk in &metadata.chunks {
            end = end
                .checked_add(chunk.pages.len())
                .ok_or_else(|| vortex_err!("Pco page count overflows usize"))?;
            chunk_ends.push(end);
        }
        Ok(Self::IndexedFast(Arc::new(IndexedIndex {
            metadata,
            chunk_ends,
        })))
    }
}

/// Shared metadata owner with only an O(chunks) page-to-chunk map; no dense-value prefixes.
#[derive(Debug)]
pub(crate) struct IndexedIndex {
    metadata: Arc<PcoMetadata>,
    chunk_ends: Vec<usize>,
}
impl IndexedIndex {
    pub fn view(&self) -> IndexedView<'_> {
        IndexedView { index: self }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct IndexedView<'a> {
    index: &'a IndexedIndex,
}
impl<'a> IndexedView<'a> {
    pub fn page_count(self) -> usize {
        self.index.chunk_ends.last().copied().unwrap_or(0)
    }
    pub fn find_row(self, row: usize) -> usize {
        self.ends().partition_point(|end| *end <= row as u64)
    }
    fn ends(self) -> &'a [u64] {
        &self
            .index
            .metadata
            .logical_row_prefix
            .as_ref()
            .vortex_expect("indexed owner has a logical-row prefix")
            .logical_row_ends
    }
    /// Unsliced logical rows, including nulls. Inter-page nulls belong to the following page;
    /// the final page includes trailing nulls. These coordinates are never slice-relative.
    pub fn rows(self, page: usize) -> VortexResult<Range<usize>> {
        let ends = self.ends();
        let start = if page == 0 {
            0
        } else {
            usize::try_from(ends[page - 1])?
        };
        Ok(start..usize::try_from(ends[page])?)
    }
    pub fn physical(self, page: usize, cursor: &mut ChunkCursor) -> VortexResult<PhysicalPage> {
        if !cursor.pages.contains(&page) {
            let chunk = self.index.chunk_ends.partition_point(|end| *end <= page);
            let end = *self
                .index
                .chunk_ends
                .get(chunk)
                .ok_or_else(|| vortex_err!("Pco physical page is missing"))?;
            let start = if chunk == 0 {
                0
            } else {
                self.index.chunk_ends[chunk - 1]
            };
            *cursor = ChunkCursor {
                chunk,
                pages: start..end,
            };
        }
        let info = self
            .index
            .metadata
            .chunks
            .get(cursor.chunk)
            .and_then(|chunk| chunk.pages.get(page - cursor.pages.start))
            .ok_or_else(|| vortex_err!("Pco page metadata is missing"))?;
        Ok(PhysicalPage {
            chunk: cursor.chunk,
            n_values: info.n_values as usize,
        })
    }
}

/// Selected physical-page decode parameters, not logical row or compressed byte offsets.
pub(crate) struct PhysicalPage {
    /// Zero-based position in metadata.chunks and chunk_metas.
    pub chunk: usize,
    /// Number of initialized primitive output slots required by this page.
    pub n_values: usize,
}
#[derive(Default)]
pub(crate) struct ChunkCursor {
    chunk: usize,
    pages: Range<usize>,
}
