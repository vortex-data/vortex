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
use vortex_utils::aliases::hash_map::HashMap;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::VERSION;
use crate::footer::FileStatistics;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptMetadata;
use crate::footer::postscript::PostscriptSegment;

/// Deserialize a footer from the end of a Vortex file or created from a
/// [`crate::footer::FooterSerializer`].
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
    // Whether to include user-defined metadata segments in the footer read.
    include_metadata: bool,
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
            include_metadata: false,
            postscript: None,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_some_dtype(mut self, dtype: Option<DType>) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    pub fn with_some_size(mut self, file_size: Option<u64>) -> Self {
        self.file_size = file_size;
        self
    }

    /// Include user-defined metadata segments in footer deserialization.
    pub fn include_metadata(mut self) -> Self {
        self.include_metadata = true;
        self
    }

    /// Whether to include user-defined metadata segments in footer deserialization.
    pub fn with_include_metadata(mut self, include_metadata: bool) -> Self {
        self.include_metadata = include_metadata;
        self
    }

    /// Prefix more data to the existing buffer when requested by the deserializer.
    pub fn prefix_data(&mut self, more_data: ByteBuffer) {
        let mut buffer = ByteBufferMut::with_capacity(self.buffer.len() + more_data.len());
        buffer.extend_from_slice(&more_data);
        buffer.extend_from_slice(&self.buffer);
        self.buffer = buffer.freeze();
    }

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
        if self.include_metadata {
            for metadata in &postscript.metadata {
                read_more_offset = read_more_offset.min(metadata.segment.offset);
            }
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
        let metadata = if self.include_metadata {
            Arc::new(
                postscript
                    .metadata
                    .iter()
                    .map(|metadata| {
                        self.parse_metadata_segment(initial_offset, &self.buffer, metadata)
                    })
                    .collect::<VortexResult<HashMap<_, _>>>()?,
            )
        } else {
            Arc::new(HashMap::default())
        };

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
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::copy_from(&initial_read[offset..offset + (segment.length as usize)]);
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
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::copy_from(&initial_read[offset..offset + (segment.length as usize)]);

        let fb = root::<vortex_flatbuffers::footer::FileStatistics>(&sliced_buffer)?;
        FileStatistics::from_flatbuffer(&fb, dtype, session)
    }

    /// Parse a user-defined metadata segment from the initial read buffer.
    fn parse_metadata_segment(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        metadata: &PostscriptMetadata,
    ) -> VortexResult<(String, ByteBuffer)> {
        let offset = usize::try_from(metadata.segment.offset - initial_offset)?;
        let length = metadata.segment.length as usize;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Metadata segment range overflowed usize"))?;

        if end > initial_read.len() {
            vortex_bail!(
                "Metadata segment {} range {}..{} out of bounds for initial read of length {}",
                metadata.key,
                offset,
                end,
                initial_read.len()
            );
        }

        Ok((
            metadata.key.clone(),
            initial_read
                .slice_unaligned(offset..end)
                .aligned(metadata.segment.alignment),
        ))
    }

    /// Parse the rest of the footer from the initial read.
    fn parse_footer(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        postscript: &Postscript,
        dtype: DType,
        file_stats: Option<FileStatistics>,
        metadata: Arc<HashMap<String, ByteBuffer>>,
    ) -> VortexResult<Footer> {
        let footer_segment = &postscript.footer;
        let footer_offset = usize::try_from(footer_segment.offset - initial_offset)?;
        let footer_bytes = FlatBuffer::copy_from(
            &initial_read[footer_offset..footer_offset + (footer_segment.length as usize)],
        );

        let layout_segment = &postscript.layout;
        let layout_offset = usize::try_from(layout_segment.offset - initial_offset)?;
        let layout_bytes = FlatBuffer::copy_from(
            &initial_read[layout_offset..layout_offset + (layout_segment.length as usize)],
        );

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

#[derive(Debug)]
pub enum DeserializeStep {
    // The offset and length of additional data needed to continue deserialization.
    NeedMoreData { offset: u64, len: usize },
    NeedFileSize,
    Done(Footer),
}
