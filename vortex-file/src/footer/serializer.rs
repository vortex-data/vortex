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

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VERSION;
use crate::footer::file_layout::FooterFlatBufferWriter;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptSegment;

/// Serializes a [`Footer`] into footer buffers and the trailing postscript/EOF marker.
pub struct FooterSerializer {
    footer: Footer,
    exclude_dtype: bool,
    offset: u64,
}

impl FooterSerializer {
    pub(super) fn new(footer: Footer) -> Self {
        Self {
            footer,
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

    /// Serialize the footer into a byte buffer that can later be deserialized as a [`Footer`].
    /// This can be helpful for storing some footer data out-of-band to accelerate opening a file.
    pub fn serialize(mut self) -> VortexResult<Vec<ByteBuffer>> {
        let mut buffers = vec![];

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

        Ok(buffers)
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
