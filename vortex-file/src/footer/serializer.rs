// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_layout::LayoutContext;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VERSION;
use crate::footer::SegmentSpec;
use crate::footer::file_layout::FooterFlatBufferWriter;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptMetadata;
use crate::footer::postscript::PostscriptSegment;

/// Serializes a [`Footer`] into footer buffers and the trailing postscript/EOF marker.
pub struct FooterSerializer {
    footer: Footer,
    metadata: HashMap<String, ByteBuffer>,
    exclude_dtype: bool,
    offset: u64,
}

impl FooterSerializer {
    pub(super) fn new(footer: Footer) -> Self {
        Self {
            footer,
            metadata: HashMap::default(),
            exclude_dtype: false,
            offset: 0,
        }
    }

    /// Update the offset used to generate absolute segment locations.
    ///
    /// This represents the byte position that the first buffer emitted by this serializer will be
    /// written to.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Exclude the DType from the serialized footer.
    /// If excluded, the reader must be provided the DType from an external source.
    pub fn exclude_dtype(mut self) -> Self {
        self.exclude_dtype = true;
        self
    }

    /// Whether to exclude the DType from the serialized footer.
    /// If excluded, the reader must be provided the DType from an external source.
    pub fn with_exclude_dtype(mut self, exclude_dtype: bool) -> Self {
        self.exclude_dtype = exclude_dtype;
        self
    }

    pub(crate) fn with_metadata_segments(mut self, metadata: HashMap<String, ByteBuffer>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Serialize the footer into a byte buffer that can later be deserialized as a [`Footer`].
    /// This can be helpful for storing some footer data out-of-band to accelerate opening a file.
    pub fn serialize(self) -> VortexResult<Vec<ByteBuffer>> {
        self.serialize_with_metadata()
            .map(|(buffers, _metadata, _approx_byte_size)| buffers)
    }

    pub(crate) fn serialize_with_metadata(
        mut self,
    ) -> VortexResult<(Vec<ByteBuffer>, super::MetadataSegments, usize)> {
        let mut buffers = vec![];

        let (metadata_segments, metadata_locators) = if self.metadata.is_empty() {
            let locators = self
                .footer
                .metadata_segments()
                .map(|(key, segment)| (key.to_string(), *segment))
                .collect::<Vec<_>>();
            let segments = locators
                .iter()
                .map(|(key, segment)| PostscriptMetadata {
                    key: key.clone(),
                    segment: PostscriptSegment::from(segment),
                })
                .collect();
            (segments, locators)
        } else {
            let mut metadata = self.metadata.into_iter().collect::<Vec<_>>();
            // Metadata is stored in a HashMap, so sort keys before assigning segment offsets.
            metadata.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

            let mut segments = Vec::with_capacity(metadata.len());
            let mut locators = Vec::with_capacity(metadata.len());
            for (key, metadata) in metadata {
                let (metadata_buffers, segment) = write_buffer(&mut self.offset, metadata)?;
                buffers.extend(metadata_buffers);
                locators.push((key.clone(), SegmentSpec::from(&segment)));
                segments.push(PostscriptMetadata { key, segment });
            }
            (segments, locators)
        };
        let metadata_storage_bytes = buffers.iter().map(ByteBuffer::len).sum::<usize>();

        let dtype_segment = if self.exclude_dtype {
            None
        } else {
            let (buffer, dtype_segment) = write_flatbuffer(&mut self.offset, self.footer.dtype())?;
            buffers.push(buffer);
            Some(dtype_segment)
        };

        // TODO(ngates): we should separate the read/write side of Context since the write side
        //  doesn't need to look anything up in the registry.
        let layout_ctx = LayoutContext::default();

        let (buffer, layout_segment) = write_flatbuffer(
            &mut self.offset,
            &self.footer.layout().flatbuffer_writer(&layout_ctx),
        )?;
        buffers.push(buffer);

        let statistics_segment = match self.footer.statistics() {
            None => None,
            Some(stats) if stats.stats_sets().is_empty() => None,
            Some(stats) => {
                let (buffer, stats_segment) = write_flatbuffer(&mut self.offset, stats)?;
                buffers.push(buffer);
                Some(stats_segment)
            }
        };

        let (buffer, footer_segment) = write_flatbuffer(
            &mut self.offset,
            &FooterFlatBufferWriter {
                ctx: self.footer.array_read_ctx.clone(),
                layout_ctx: ReadContext::new(layout_ctx.to_ids()),
                segment_specs: Arc::clone(&self.footer.segments),
            },
        )?;
        buffers.push(buffer);

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
            layout: layout_segment,
            statistics: statistics_segment,
            footer: footer_segment,
            metadata: metadata_segments,
        };
        let postscript_buffer = postscript.write_flatbuffer_bytes()?;
        if postscript_buffer.len() > MAX_POSTSCRIPT_SIZE as usize {
            Err(vortex_err!(
                "Postscript is too large ({} bytes); max postscript size is {}",
                postscript_buffer.len(),
                MAX_POSTSCRIPT_SIZE
            ))?;
        }

        let postscript_len = u16::try_from(postscript_buffer.len())
            .vortex_expect("Postscript already verified to fit into u16");
        buffers.push(postscript_buffer.into_inner());

        // And finally, the EOF 8-byte footer.
        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        buffers.push(ByteBuffer::copy_from(eof));

        let approx_byte_size =
            buffers.iter().map(ByteBuffer::len).sum::<usize>() - metadata_storage_bytes;
        Ok((buffers, metadata_locators.into(), approx_byte_size))
    }
}

impl From<&PostscriptSegment> for SegmentSpec {
    fn from(value: &PostscriptSegment) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
            alignment: value.alignment,
        }
    }
}

impl From<&SegmentSpec> for PostscriptSegment {
    fn from(value: &SegmentSpec) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
            alignment: value.alignment,
        }
    }
}

fn write_flatbuffer<F: FlatBufferRoot + WriteFlatBuffer>(
    offset: &mut u64,
    flatbuffer: &F,
) -> VortexResult<(ByteBuffer, PostscriptSegment)> {
    let buffer = flatbuffer.write_flatbuffer_bytes()?;
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("flatbuffer length exceeds maximum u32"))?;

    let segment = PostscriptSegment {
        offset: *offset,
        length,
        alignment: FlatBuffer::alignment(),
    };

    *offset += u64::from(length);

    Ok((buffer.into_inner(), segment))
}

fn write_buffer(
    offset: &mut u64,
    buffer: ByteBuffer,
) -> VortexResult<(Vec<ByteBuffer>, PostscriptSegment)> {
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("metadata segment length exceeds maximum u32"))?;
    let alignment = buffer.alignment();

    let padding = offset.next_multiple_of(*alignment as u64) - *offset;
    let segment_offset = *offset + padding;

    let segment = PostscriptSegment {
        offset: segment_offset,
        length,
        alignment,
    };

    *offset += padding + u64::from(length);

    let mut buffers = Vec::with_capacity(if padding == 0 { 1 } else { 2 });
    if padding > 0 {
        buffers.push(ByteBuffer::zeroed(padding as usize));
    }
    buffers.push(buffer);

    Ok((buffers, segment))
}
