// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the footer of a Vortex file, which contains metadata about the file's contents.
//!
//! The footer includes:
//! - The file's layout, which describes how the data is organized
//! - Statistics about the data, which can be used for query optimization
//! - Segment map, which describe the physical location of data in the file
//!
//! The footer is located at the end of the file and is used to interpret the file's contents.
mod file_layout;
mod file_statistics;
mod postscript;
mod segment;

use std::sync::Arc;

mod serializer;
pub use serializer::*;
mod deserializer;
pub use deserializer::*;
pub use file_statistics::FileStatistics;
use flatbuffers::root;
use itertools::Itertools;
pub use segment::*;
use vortex_array::ArrayId;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::footer as fb;
use vortex_layout::LayoutEncodingId;
use vortex_layout::LayoutRef;
use vortex_layout::layout_from_flatbuffer_with_options;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    root_layout: LayoutRef,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
    // The specific arrays used within the file, in the order they were registered.
    array_read_ctx: ReadContext,
    // The approximate size of the footer in bytes, used for caching and memory management.
    approx_byte_size: Option<usize>,
}

impl Footer {
    pub(crate) fn new(
        root_layout: LayoutRef,
        segments: Arc<[SegmentSpec]>,
        statistics: Option<FileStatistics>,
        array_read_ctx: ReadContext,
    ) -> Self {
        Self {
            root_layout,
            segments,
            statistics,
            array_read_ctx,
            approx_byte_size: None,
        }
    }

    pub(crate) fn with_approx_byte_size(mut self, approx_byte_size: usize) -> Self {
        self.approx_byte_size = Some(approx_byte_size);
        self
    }

    /// Read the [`Footer`] from a flatbuffer.
    pub(crate) fn from_flatbuffer(
        footer_bytes: FlatBuffer,
        layout_bytes: FlatBuffer,
        dtype: DType,
        statistics: Option<FileStatistics>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let approx_byte_size = footer_bytes.len() + layout_bytes.len();
        let fb_footer = root::<fb::Footer>(&footer_bytes)?;

        // Create a LayoutContext from the registry.
        let layout_specs = fb_footer.layout_specs();
        let layout_ids: Arc<[_]> = layout_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| LayoutEncodingId::new(encoding.id()))
            .collect();
        let layout_read_ctx = ReadContext::new(layout_ids);

        // Create an ArrayContext from the registry.
        let array_specs = fb_footer.array_specs();
        let array_ids: Arc<[_]> = array_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| ArrayId::new(encoding.id()))
            .collect();
        let array_read_ctx = ReadContext::new(array_ids);

        let root_layout = layout_from_flatbuffer_with_options(
            layout_bytes,
            &dtype,
            &layout_read_ctx,
            &array_read_ctx,
            session.layouts().registry(),
            session.allows_unknown(),
        )?;

        let segments: Arc<[SegmentSpec]> = fb_footer
            .segment_specs()
            .ok_or_else(|| vortex_err!("FileLayout missing segment specs"))?
            .iter()
            .map(SegmentSpec::try_from)
            .try_collect()?;

        // Note this assertion is `<=` since we allow zero-length segments
        if !segments.is_sorted_by_key(|segment| segment.offset) {
            vortex_bail!("Segment offsets are not ordered");
        }

        Ok(Self {
            root_layout,
            segments,
            statistics,
            array_read_ctx,
            approx_byte_size: Some(approx_byte_size),
        })
    }

    /// Returns the root [`LayoutRef`] of the file.
    pub fn layout(&self) -> &LayoutRef {
        &self.root_layout
    }

    /// Returns the segment map of the file.
    pub fn segment_map(&self) -> &Arc<[SegmentSpec]> {
        &self.segments
    }

    /// Returns the statistics of the file.
    pub fn statistics(&self) -> Option<&FileStatistics> {
        self.statistics.as_ref()
    }

    /// Returns the [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        self.root_layout.dtype()
    }

    /// Returns the approximate size of the footer in bytes, used for caching and memory management.
    pub fn approx_byte_size(&self) -> Option<usize> {
        self.approx_byte_size
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.root_layout.row_count()
    }

    /// Returns a serializer for this footer.
    pub fn into_serializer(self) -> FooterSerializer {
        FooterSerializer::new(self)
    }

    /// Create a deserializer for a Vortex file footer.
    pub fn deserializer(eof_buffer: ByteBuffer, session: VortexSession) -> FooterDeserializer {
        FooterDeserializer::new(eof_buffer, session)
    }
}
