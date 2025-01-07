use std::io::Read;

use flatbuffers::root;
use itertools::Itertools;
use vortex_array::ContextRef;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::{dtype as fbd, footer2 as fb, ReadFlatBuffer};
use vortex_io::VortexReadAt;
use vortex_layout::segments::SegmentId;
use vortex_layout::{LayoutContextRef, LayoutData, LayoutId};

use crate::v2::footer::{FileLayout, Postscript, Segment};
use crate::v2::segments::SegmentCache;
use crate::v2::VortexFile;
use crate::{EOF_SIZE, MAGIC_BYTES, VERSION};

const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

/// Open options for a Vortex file reader.
pub struct OpenOptions {
    /// The Vortex Array encoding context.
    ctx: ContextRef,
    /// The Vortex Layout encoding context.
    layout_ctx: LayoutContextRef,
    /// An optional, externally provided, file layout.
    file_layout: Option<FileLayout>,
    /// An optional, externally provided, dtype.
    dtype: Option<DType>,
    // TODO(ngates): also support a messages_middleware that can wrap a message cache to provide
    //  additional caching, metrics, or other intercepts, etc. It should support synchronous
    //  read + write of Map<MessageId, ByteBuffer> or similar.
    initial_read_size: u64,
}

impl OpenOptions {
    pub fn new(ctx: ContextRef) -> Self {
        Self {
            ctx,
            layout_ctx: LayoutContextRef::default(),
            file_layout: None,
            dtype: None,
            initial_read_size: INITIAL_READ_SIZE,
        }
    }

    /// Configure the initial read size for the Vortex file.
    pub fn with_initial_read_size(mut self, initial_read_size: u64) -> VortexResult<Self> {
        if self.initial_read_size < u16::MAX as u64 {
            vortex_bail!("initial_read_size must be at least u16::MAX");
        }
        self.initial_read_size = initial_read_size;
        Ok(self)
    }
}

impl OpenOptions {
    /// Open the Vortex file using synchronous IO.
    pub fn open_sync<R: Read>(self, _read: R) -> VortexResult<VortexFile<R>> {
        todo!()
    }

    /// Open the Vortex file using asynchronous IO.
    pub async fn open<R: VortexReadAt>(self, read: R) -> VortexResult<VortexFile<R>> {
        // Fetch the file size and perform the initial read.
        let file_size = read.size().await?;
        let initial_read_size = self.initial_read_size.min(file_size);
        let initial_offset = file_size - initial_read_size;
        let initial_read: ByteBuffer = read
            .read_byte_range(initial_offset, initial_read_size)
            .await?
            .into();

        // We know the initial read _must_ contain at least the Postscript.
        let postscript = self.parse_postscript(&initial_read)?;

        // Check if we need to read more bytes.
        // NOTE(ngates): for now, we assume the dtype and layout segments are adjacent.
        let (initial_offset, initial_read) = if (self.dtype.is_none()
            && postscript.dtype.offset < initial_offset)
            || (self.file_layout.is_none() && postscript.file_layout.offset < initial_offset)
        {
            let offset = postscript.dtype.offset.min(postscript.file_layout.offset);
            let mut new_initial_read =
                ByteBufferMut::with_capacity(usize::try_from(file_size - offset)?);
            new_initial_read.extend_from_slice(
                &read
                    .read_byte_range(offset, initial_offset - offset)
                    .await?,
            );
            new_initial_read.extend_from_slice(&initial_read);
            (offset, new_initial_read.freeze())
        } else {
            (initial_offset, initial_read)
        };

        // Now we try to read the DType and Layout segments.
        let dtype = self.dtype.clone().unwrap_or_else(|| {
            self.parse_dtype(initial_offset, &initial_read, postscript.dtype)
                .vortex_expect("Failed to parse dtype")
        });
        let file_layout = self.file_layout.clone().unwrap_or_else(|| {
            self.parse_file_layout(
                initial_offset,
                &initial_read,
                postscript.file_layout,
                dtype.clone(),
            )
            .vortex_expect("Failed to parse file layout")
        });

        // Set up our segment cache and for good measure, we populate any segments that were
        // covered by the initial read.
        let mut segment_cache = SegmentCache::default();
        self.populate_segments(
            initial_offset,
            &initial_read,
            &file_layout,
            &mut segment_cache,
        )?;

        // Finally, create the VortexFile.
        Ok(VortexFile {
            read,
            ctx: self.ctx.clone(),
            layout: file_layout.root_layout,
            segments: file_layout.segments,
            segment_cache,
        })
    }

    /// Parse the postscript from the initial read.
    fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
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

        Postscript::read_flatbuffer_bytes(&initial_read[eof_loc - ps_size..eof_loc])
    }

    /// Parse the DType from the initial read.
    fn parse_dtype(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        dtype: Segment,
    ) -> VortexResult<DType> {
        let offset = usize::try_from(dtype.offset - initial_offset)?;
        let dtype_bytes = &initial_read[offset..offset + dtype.length];
        DType::try_from(root::<fbd::DType>(dtype_bytes)?)
    }

    /// Parse the FileLayout from the initial read.
    fn parse_file_layout(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        segment: Segment,
        dtype: DType,
    ) -> VortexResult<FileLayout> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let bytes = initial_read.slice(offset..offset + segment.length);

        let fb = root::<fb::FileLayout>(&bytes)?;
        let fb_root_layout = fb
            .root_layout()
            .ok_or_else(|| vortex_err!("FileLayout missing root layout"))?;

        let root_encoding = self
            .layout_ctx
            .lookup_layout(LayoutId(fb_root_layout.encoding()))
            .ok_or_else(|| {
                vortex_err!(
                    "FileLayout root layout encoding {} not found",
                    fb_root_layout.encoding()
                )
            })?;
        let _fb_encoding_id = fb_root_layout.encoding();
        let root_layout = LayoutData::try_new_viewed(
            root_encoding,
            dtype,
            bytes.clone(),
            fb_root_layout._tab.loc(),
            self.layout_ctx.clone(),
        )?;

        let fb_segments = fb
            .segments()
            .ok_or_else(|| vortex_err!("FileLayout missing segments"))?;
        let segments = fb_segments
            .iter()
            .map(|s| Segment::read_flatbuffer(&s))
            .try_collect()?;

        Ok(FileLayout {
            root_layout,
            segments,
        })
    }

    fn populate_segments(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        file_layout: &FileLayout,
        segments: &mut SegmentCache,
    ) -> VortexResult<()> {
        for (idx, segment) in file_layout.segments.iter().enumerate() {
            if segment.offset < initial_offset {
                // Skip segments that aren't in the initial read.
                continue;
            }

            let segment_id = SegmentId::from(u32::try_from(idx)?);

            let offset = usize::try_from(segment.offset - initial_offset)?;
            let bytes = initial_read.slice(offset..offset + segment.length);

            segments.set(segment_id, bytes.into_inner());
        }
        Ok(())
    }
}
