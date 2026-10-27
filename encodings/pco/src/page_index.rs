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
    IndexedFast(Arc<LogicalPageIndex>),
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
        Ok(Self::IndexedFast(Arc::new(LogicalPageIndex {
            metadata,
            chunk_ends,
        })))
    }
}

/// Shared metadata owner with only an O(chunks) page-to-chunk map; no dense-value prefixes.
#[derive(Debug)]
pub(crate) struct LogicalPageIndex {
    metadata: Arc<PcoMetadata>,
    chunk_ends: Vec<usize>,
}
impl LogicalPageIndex {
    pub fn page_count(&self) -> usize {
        self.chunk_ends.last().copied().unwrap_or(0)
    }
    pub fn find_row(&self, row: usize) -> usize {
        self.ends().partition_point(|end| *end <= row as u64)
    }
    fn ends(&self) -> &[u64] {
        &self
            .metadata
            .logical_row_prefix
            .as_ref()
            .vortex_expect("indexed owner has a logical-row prefix")
            .logical_row_ends
    }
    /// Unsliced logical rows, including nulls. Inter-page nulls belong to the following page;
    /// the final page includes trailing nulls. These coordinates are never slice-relative.
    pub fn rows(&self, page: usize) -> Range<u64> {
        let ends = self.ends();
        let start = if page == 0 { 0 } else { ends[page - 1] };
        start..ends[page]
    }
    pub fn physical(&self, page: usize) -> VortexResult<PhysicalPage> {
        let chunk = self.chunk_ends.partition_point(|end| *end <= page);
        let start = if chunk == 0 {
            0
        } else {
            self.chunk_ends[chunk - 1]
        };
        let info = self
            .metadata
            .chunks
            .get(chunk)
            .and_then(|chunk| chunk.pages.get(page - start))
            .ok_or_else(|| vortex_err!("Pco page metadata is missing"))?;
        Ok(PhysicalPage {
            chunk,
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
