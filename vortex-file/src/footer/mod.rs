// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex file footer metadata.
//!
//! A footer contains the root layout, file-level statistics, the segment map, and the read contexts
//! needed to resolve array/layout encoding ids during deserialization.
//!
//! The byte-level footer and postscript layout is part of the file-format spec; this module exposes
//! the structured Rust representation and serializer/deserializer state machine.
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
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

/// Maximum number of user-defined metadata segments. Keeps postscript bookkeeping small so the
/// footer and required segments still fit the initial tail read.
pub(crate) const MAX_METADATA_SEGMENTS: usize = 16;

/// Maximum length, in UTF-8 bytes, of a user-defined metadata key (keys live in the postscript).
pub(crate) const MAX_METADATA_KEY_BYTES: usize = 32;

/// User-defined metadata segment locators stored as `(key, locator)` pairs.
pub(crate) type MetadataSegments = Arc<[(String, SegmentSpec)]>;

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    root_layout: LayoutRef,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
    metadata: Arc<[(String, SegmentSpec)]>,
    // The specific arrays used within the file, in the order they were registered.
    array_read_ctx: ReadContext,
    // The approximate size of the footer in bytes, used for caching and memory management.
    approx_byte_size: Option<usize>,
}

impl Footer {
    pub fn new(
        root_layout: LayoutRef,
        segments: Arc<[SegmentSpec]>,
        statistics: Option<FileStatistics>,
        array_read_ctx: ReadContext,
    ) -> Self {
        Self {
            root_layout,
            segments,
            statistics,
            metadata: Arc::from([]),
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
        metadata: Arc<[(String, SegmentSpec)]>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let metadata_bytes: usize = metadata
            .iter()
            .map(|(key, _segment)| key.len() + size_of::<SegmentSpec>())
            .sum();
        let approx_byte_size = footer_bytes.len() + layout_bytes.len() + metadata_bytes;
        let fb_footer = root::<fb::Footer>(&footer_bytes)?;

        // Create a LayoutContext from the registry.
        let layout_specs = fb_footer.layout_specs();
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let layout_ids: Arc<[_]> = layout_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| LayoutEncodingId::new(encoding.id()))
            .collect();
        let layout_read_ctx = ReadContext::new(layout_ids);

        // Create an ArrayContext from the registry.
        let array_specs = fb_footer.array_specs();
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
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
            session,
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
            metadata,
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

    /// Returns the user-defined metadata segment locators stored in the postscript.
    pub fn metadata_segments(&self) -> impl Iterator<Item = (&str, &SegmentSpec)> {
        self.metadata
            .iter()
            .map(|(key, segment)| (key.as_str(), segment))
    }

    /// Returns the user-defined metadata segment locator for the given key.
    pub fn metadata_segment(&self, key: &str) -> Option<&SegmentSpec> {
        self.metadata
            .iter()
            .find_map(|(candidate, segment)| (candidate == key).then_some(segment))
    }

    pub(crate) fn with_metadata_segments(mut self, metadata: Arc<[(String, SegmentSpec)]>) -> Self {
        self.metadata = metadata;
        self
    }

    pub(crate) fn segment_specs_with_metadata(&self) -> Arc<[SegmentSpec]> {
        self.segments
            .iter()
            .copied()
            .chain(self.metadata.iter().map(|(_, segment)| *segment))
            .collect()
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
