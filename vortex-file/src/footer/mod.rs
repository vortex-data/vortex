// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex file footer metadata.
//!
//! A footer contains the root layout, file-level statistics, the segment map, and the read contexts
//! needed to resolve array/layout encoding ids during deserialization.
//!
//! The byte-level footer and postscript layout is part of the file-format spec; this module exposes
//! the structured Rust representation and serializer/deserializer state machine.
mod field_sizes;
mod file_layout;
mod file_statistics;
mod postscript;
mod segment;

use std::sync::Arc;

mod serializer;
pub use serializer::*;
mod deserializer;
pub use deserializer::*;
pub use field_sizes::CompressedFieldSizes;
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
            array_read_ctx,
            approx_byte_size: None,
        }
    }

    pub(crate) fn with_approx_byte_size(mut self, approx_byte_size: usize) -> Self {
        self.approx_byte_size = Some(approx_byte_size);
        self
    }

    /// Read the [`Footer`] from a flatbuffer.
    ///
    /// `file_size` is the total size of the file in bytes and is used to validate that every
    /// segment declared in the footer lies within the file.
    pub(crate) fn from_flatbuffer(
        footer_bytes: &[u8],
        layout_bytes: FlatBuffer,
        dtype: DType,
        statistics: Option<FileStatistics>,
        file_size: u64,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let approx_byte_size = footer_bytes.len() + layout_bytes.len();
        let fb_footer = root::<fb::Footer>(footer_bytes)?;

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

        // A corrupt or malicious file can declare a segment whose offset or length extends past
        // the end of the file. Reject such files here so that later slicing of the backing buffer
        // returns a `VortexError` rather than panicking (see issue #8819).
        validate_segments_within_file(&segments, file_size)?;

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

    /// Computes the compressed size in bytes of every field in the file, keyed by field path.
    ///
    /// Sizes are derived by attributing each segment in the [segment map][Self::segment_map] to a
    /// field in the [layout tree][Self::layout]; see [`CompressedFieldSizes`] for the exact
    /// attribution semantics. No IO is performed.
    pub fn compressed_field_sizes(&self) -> VortexResult<CompressedFieldSizes> {
        CompressedFieldSizes::try_new(&self.root_layout, &self.segments)
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

/// Validate that every segment declared in the footer lies within a file of `file_size` bytes.
///
/// A corrupt or malicious file can declare a segment whose offset or length extends past the end
/// of the file. Rejecting such files up front ensures that later slicing of the backing buffer
/// returns a [`VortexError`](vortex_error::VortexError) rather than panicking (see issue #8819).
fn validate_segments_within_file(segments: &[SegmentSpec], file_size: u64) -> VortexResult<()> {
    for segment in segments {
        let within_file = segment
            .offset
            .checked_add(segment.length as u64)
            .is_some_and(|end| end <= file_size);
        if !within_file {
            vortex_bail!(
                "Segment at offset {} with length {} extends past the end of the \
                 {file_size}-byte file",
                segment.offset,
                segment.length,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Alignment;

    use super::*;

    fn segment(offset: u64, length: u32) -> SegmentSpec {
        SegmentSpec {
            offset,
            length,
            alignment: Alignment::none(),
        }
    }

    #[test]
    fn accepts_segments_within_file() -> VortexResult<()> {
        validate_segments_within_file(&[segment(0, 100), segment(100, 50)], 150)?;
        Ok(())
    }

    #[test]
    fn rejects_segment_extending_past_end_of_file() {
        let err =
            validate_segments_within_file(&[segment(0, 100), segment(100, 51)], 150).unwrap_err();
        assert!(err.to_string().contains("past the end"), "{err}");
    }

    #[test]
    fn rejects_segment_offset_length_overflow() {
        let err = validate_segments_within_file(&[segment(u64::MAX, 1)], u64::MAX).unwrap_err();
        assert!(err.to_string().contains("past the end"), "{err}");
    }
}
