// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use flatbuffers::root;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::ReadFlatBuffer;
use vortex_session::VortexSession;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::VERSION;
use crate::footer::FileStatistics;
use crate::footer::SegmentSpec;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptSegment;

/// Deserialize a footer from the end of a Vortex file or created from a
/// [`crate::footer::FooterSerializer`].
///
/// The deserializer is incremental because callers may initially read only the tail of a file. Call
/// [`deserialize`](Self::deserialize) until it returns [`DeserializeStep::Done`]. If it asks for
/// [`DeserializeStep::NeedMoreData`], prefix the requested bytes with [`prefix_data`](Self::prefix_data).
/// If it asks for [`DeserializeStep::NeedFileSize`], call [`with_size`](Self::with_size) and retry.
pub struct FooterDeserializer {
    // A buffer representing the end of a Vortex file.
    // During deserialization, we may need to expand this buffer by requesting more data from
    // the caller.
    buffer: ByteBuffer,
    // The session to use for deserialization.
    session: VortexSession,
    // The DType, if provided externally.
    dtype: Option<DType>,

    // Internal state that we accumulate

    // The file size, possibly provided externally.
    file_size: Option<u64>,
    // The postscript, once we've parsed it.
    postscript: Option<Postscript>,
}

impl FooterDeserializer {
    pub(super) fn new(initial_read: ByteBuffer, session: VortexSession) -> Self {
        Self {
            buffer: initial_read,
            session,
            dtype: None,
            file_size: None,
            postscript: None,
        }
    }

    /// Provide the file dtype externally.
    ///
    /// This is required for files written with [`VortexWriteOptions::exclude_dtype`](crate::VortexWriteOptions::exclude_dtype).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Provide or clear the externally known file dtype.
    pub fn with_some_dtype(mut self, dtype: Option<DType>) -> Self {
        self.dtype = dtype;
        self
    }

    /// Provide the total file size.
    pub fn with_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    /// Provide or clear the total file size.
    pub fn with_some_size(mut self, file_size: Option<u64>) -> Self {
        self.file_size = file_size;
        self
    }

    /// Prefix more data to the existing buffer when requested by the deserializer.
    pub fn prefix_data(&mut self, more_data: ByteBuffer) {
        let mut buffer = ByteBufferMut::with_capacity(self.buffer.len() + more_data.len());
        buffer.extend_from_slice(&more_data);
        buffer.extend_from_slice(&self.buffer);
        self.buffer = buffer.freeze();
    }

    /// Advance footer deserialization.
    ///
    /// Returns the next missing input requirement or the finished [`Footer`].
    pub fn deserialize(&mut self) -> VortexResult<DeserializeStep> {
        let postscript = if let Some(postscript) = &self.postscript {
            postscript
        } else {
            self.postscript = Some(self.parse_postscript(&self.buffer)?);
            self.postscript
                .as_ref()
                .vortex_expect("Just set postscript")
        };

        // If we haven't been provided a DType, we must read one from the file.
        let dtype_segment = self
            .dtype
            .is_none()
            .then(|| {
                postscript.dtype.as_ref().ok_or_else(|| {
                    vortex_err!(
                        "Vortex file doesn't embed a DType and none provided to VortexOpenOptions"
                    )
                })
            })
            .transpose()?;

        // The other postscript segments are required, so now we figure out our the offset that
        // contains all the required segments.

        // The initial offset is the file size - the size of our initial read.
        let Some(file_size) = self.file_size else {
            return Ok(DeserializeStep::NeedFileSize);
        };
        let initial_offset = file_size - (self.buffer.len() as u64);

        let mut read_more_offset = initial_offset;
        if let Some(dtype_segment) = &dtype_segment {
            read_more_offset = read_more_offset.min(dtype_segment.offset);
        }
        if let Some(stats_segment) = &postscript.statistics {
            read_more_offset = read_more_offset.min(stats_segment.offset);
        }
        read_more_offset = read_more_offset.min(postscript.layout.offset);
        read_more_offset = read_more_offset.min(postscript.footer.offset);

        // Read more bytes if necessary.
        if read_more_offset < initial_offset {
            tracing::trace!(
                "Initial read from {initial_offset} did not cover all footer segments, reading from {read_more_offset}"
            );
            return Ok(DeserializeStep::NeedMoreData {
                offset: read_more_offset,
                len: usize::try_from(initial_offset - read_more_offset)?,
            });
        }

        // Now we read our initial segments.
        let dtype = dtype_segment
            .map(|segment| self.parse_dtype(initial_offset, &self.buffer, segment))
            .transpose()?
            .unwrap_or_else(|| self.dtype.clone().vortex_expect("DType was provided"));
        let file_stats = postscript
            .statistics
            .as_ref()
            .map(|segment| {
                self.parse_file_statistics(
                    initial_offset,
                    &self.buffer,
                    segment,
                    &dtype,
                    &self.session,
                )
            })
            .transpose()?;
        let metadata = postscript
            .metadata
            .iter()
            .map(|metadata| {
                let segment = SegmentSpec {
                    offset: metadata.segment.offset,
                    length: metadata.segment.length,
                    alignment: metadata.segment.alignment,
                };
                let end = segment
                    .offset
                    .checked_add(u64::from(segment.length))
                    .ok_or_else(|| {
                        vortex_err!("Metadata segment {} range overflowed u64", metadata.key)
                    })?;
                if end > file_size {
                    vortex_bail!(
                        "Metadata segment {} range {}..{} exceeds file size {}",
                        metadata.key,
                        segment.offset,
                        end,
                        file_size
                    );
                }
                let offset = usize::try_from(segment.offset)?;
                if !segment.alignment.is_offset_aligned(offset) {
                    vortex_bail!(
                        "Metadata segment {} offset {} is not aligned to {}",
                        metadata.key,
                        segment.offset,
                        segment.alignment
                    );
                }
                Ok((metadata.key.clone(), segment))
            })
            .collect::<VortexResult<Arc<[_]>>>()?;

        Ok(DeserializeStep::Done(self.parse_footer(
            initial_offset,
            &self.buffer,
            postscript,
            dtype,
            file_stats,
            metadata,
        )?))
    }

    /// The current buffer being used for deserialization.
    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    /// Parse the postscript from the initial read.
    fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
        if initial_read.len() < EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least EOF_SIZE ({}) bytes",
                EOF_SIZE
            );
        }
        let eof_loc = initial_read.len() - EOF_SIZE;
        let magic_bytes_loc = eof_loc + (EOF_SIZE - MAGIC_BYTES.len());

        let magic_number = &initial_read[magic_bytes_loc..];
        if magic_number != MAGIC_BYTES {
            vortex_bail!("Malformed file, invalid magic bytes, got {magic_number:?}")
        }

        let version = u16::from_le_bytes(
            initial_read[eof_loc..eof_loc + 2]
                .try_into()
                .map_err(|e| vortex_err!("Version was not a u16 {e}"))?,
        );
        if version != VERSION {
            vortex_bail!("Malformed file, unsupported version {version}")
        }

        let ps_size = u16::from_le_bytes(
            initial_read[eof_loc + 2..eof_loc + 4]
                .try_into()
                .map_err(|e| vortex_err!("Postscript size was not a u16 {e}"))?,
        ) as usize;

        if initial_read.len() < ps_size + EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least {} bytes to include the Postscript",
                ps_size + EOF_SIZE
            );
        }

        Postscript::read_flatbuffer_bytes(&initial_read[eof_loc - ps_size..eof_loc])
    }

    /// Parse the DType from the initial read.
    fn parse_dtype(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        segment: &PostscriptSegment,
    ) -> VortexResult<DType> {
        let sliced_buffer = FlatBuffer::copy_from(checked_segment_slice(
            initial_read,
            initial_offset,
            segment,
        )?);
        DType::from_flatbuffer(sliced_buffer, &self.session)
    }

    /// Parse the [`FileStatistics`] from the initial read buffer.
    fn parse_file_statistics(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        segment: &PostscriptSegment,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<FileStatistics> {
        let sliced_buffer = checked_segment_slice(initial_read, initial_offset, segment)?;

        let fb = root::<vortex_flatbuffers::footer::FileStatistics>(sliced_buffer)?;
        FileStatistics::from_flatbuffer(&fb, dtype, session)
    }

    /// Parse the rest of the footer from the initial read.
    fn parse_footer(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        postscript: &Postscript,
        dtype: DType,
        file_stats: Option<FileStatistics>,
        metadata: Arc<[(String, SegmentSpec)]>,
    ) -> VortexResult<Footer> {
        let footer_segment = &postscript.footer;
        let footer_bytes = checked_segment_slice(initial_read, initial_offset, footer_segment)?;

        let layout_segment = &postscript.layout;
        let layout_bytes = FlatBuffer::copy_from(checked_segment_slice(
            initial_read,
            initial_offset,
            layout_segment,
        )?);

        Footer::from_flatbuffer(
            footer_bytes,
            layout_bytes,
            dtype,
            file_stats,
            metadata,
            &self.session,
        )
    }
}

fn checked_segment_slice<'a>(
    read: &'a [u8],
    read_offset: u64,
    segment: &PostscriptSegment,
) -> VortexResult<&'a [u8]> {
    let offset = usize::try_from(segment.offset.checked_sub(read_offset).ok_or_else(|| {
        vortex_err!(
            "Segment offset {} is smaller than file read offset {read_offset}",
            segment.offset
        )
    })?)?;
    offset
        .checked_add(segment.length as usize)
        .and_then(|end| read.get(offset..end))
        .ok_or_else(|| {
            vortex_err!(
                "Segment length {} (at offset {}) out of bounds of slice of length {}",
                segment.length,
                offset,
                read.len()
            )
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array_session;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_flatbuffers::WriteFlatBufferExt;

    use super::*;

    fn segment(offset: u64, length: u32) -> PostscriptSegment {
        PostscriptSegment {
            offset,
            length,
            alignment: FlatBuffer::alignment(),
        }
    }

    #[test]
    fn in_bounds_segment_slice() -> VortexResult<()> {
        let read: Vec<u8> = (0u8..10).collect();
        let sliced = checked_segment_slice(&read, 100, &segment(104, 4))?;
        assert_eq!(sliced, &read[4..8]);
        Ok(())
    }

    #[rstest]
    #[case::offset_before_read_start(100, 99, 4, "smaller than file read offset")]
    #[case::end_past_buffer(100, 105, 6, "out of bounds")]
    #[case::offset_past_buffer(100, 120, 1, "out of bounds")]
    #[case::end_overflows_usize(0, u64::MAX, u32::MAX, "out of bounds")]
    fn out_of_bounds_segment_slice(
        #[case] read_offset: u64,
        #[case] segment_offset: u64,
        #[case] segment_length: u32,
        #[case] expected: &str,
    ) {
        let read = [0u8; 10];
        let err =
            checked_segment_slice(&read, read_offset, &segment(segment_offset, segment_length))
                .unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    }

    fn eof_buffer(postscript: &Postscript) -> VortexResult<ByteBuffer> {
        let postscript_bytes = postscript.write_flatbuffer_bytes()?;
        let mut buffer = ByteBufferMut::with_capacity(postscript_bytes.len() + EOF_SIZE);
        buffer.extend_from_slice(&postscript_bytes);
        buffer.extend_from_slice(&VERSION.to_le_bytes());
        buffer.extend_from_slice(&u16::try_from(postscript_bytes.len())?.to_le_bytes());
        buffer.extend_from_slice(&MAGIC_BYTES);
        Ok(buffer.freeze())
    }

    #[rstest]
    #[case::length_past_eof(segment(0, u32::MAX))]
    #[case::offset_overflow(segment(u64::MAX, u32::MAX))]
    fn deserialize_rejects_out_of_bounds_footer_segment(
        #[case] footer_segment: PostscriptSegment,
    ) -> VortexResult<()> {
        let postscript = Postscript {
            dtype: None,
            layout: segment(0, 1),
            statistics: None,
            footer: footer_segment,
            metadata: Vec::new(),
        };
        let buffer = eof_buffer(&postscript)?;
        let file_size = buffer.len() as u64;

        let mut deserializer = FooterDeserializer::new(buffer, array_session())
            .with_dtype(DType::Primitive(PType::I32, Nullability::NonNullable))
            .with_size(file_size);
        let err = deserializer.deserialize().unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");
        Ok(())
    }
}

#[derive(Debug)]
/// Result of one [`FooterDeserializer::deserialize`] step.
pub enum DeserializeStep {
    /// Additional data needed to continue deserialization.
    NeedMoreData {
        /// Absolute file offset to read from.
        offset: u64,
        /// Number of bytes to read and prefix into the deserializer.
        len: usize,
    },
    /// The total file size is required before offsets can be resolved.
    NeedFileSize,
    /// Footer deserialization is complete.
    Done(Footer),
}
