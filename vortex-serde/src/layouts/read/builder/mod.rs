use std::sync::{Arc, RwLock};

use bytes::BytesMut;
use footer::read_initial_bytes;
use initial_read::read_initial_bytes;
use vortex_array::{Array, ArrayDType};
use vortex_error::VortexResult;
use vortex_expr::Select;
use vortex_schema::projection::Projection;

use super::RowMask;
use crate::io::VortexReadAt;
use crate::layouts::read::cache::{LayoutMessageCache, RelativeLayoutCache};
use crate::layouts::read::context::LayoutDeserializer;
use crate::layouts::read::filtering::RowFilter;
use crate::layouts::read::stream::LayoutBatchStream;
use crate::layouts::read::Scan;

mod footer;
mod initial_read;

/// Builder for reading Vortex files.
///
/// Succinctly, the file format specification is as follows:
///
/// 1. Data messages are written first, followed by...
/// 2. An optional Schema, which if present is a valid DType flatbuffer, and is followed by...
/// 3. The Layout, which is a valid Layout flatbuffer, and is followed by...
/// 4. The Footer, which is a valid Footer flatbuffer, and is followed by...
/// 5. The End-of-File marker, which is 8 bytes, and contains the u16 version, u16 footer length, and 4 magic bytes.
///
/// # File Format
/// ```text
/// ┌────────────────────────────┐
/// │                            │
/// │            Data            │
/// │    (Array IPC Messages)    │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │   Per-Column Statistics    │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │          Schema            │
/// |     (DType Flatbuffer)     │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │     Layout Flatbuffer      │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │     Footer Flatbuffer      │
/// │  (Schema & Layout Offsets) │
/// │                            │
/// ├────────────────────────────┤
/// │     8-byte End of File     │
/// │  (Version, Footer Length,  │
/// │       Magic Bytes)         │
/// └────────────────────────────┘
/// ```
pub struct VortexReadBuilder<R> {
    read_at: R,
    layout_serde: LayoutDeserializer,
    projection: Option<Projection>,
    size: Option<u64>,
    row_mask: Option<Array>,
    row_filter: Option<RowFilter>,
}

impl<R: VortexReadAt> VortexReadBuilder<R> {
    pub fn new(read_at: R, layout_serde: LayoutDeserializer) -> Self {
        Self {
            read_at,
            layout_serde,
            projection: None,
            size: None,
            row_mask: None,
            row_filter: None,
        }
    }

    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }

    pub fn with_projection(mut self, projection: Projection) -> Self {
        self.projection = Some(projection);
        self
    }

    pub fn with_indices(mut self, array: Array) -> Self {
        assert!(
            !array.dtype().is_nullable() && (array.dtype().is_int() || array.dtype().is_boolean()),
            "Mask arrays have to be non-nullable integer or boolean arrays"
        );

        self.row_mask = Some(array);
        self
    }

    pub fn with_row_filter(mut self, row_filter: RowFilter) -> Self {
        self.row_filter = Some(row_filter);
        self
    }

    pub async fn build(self) -> VortexResult<LayoutBatchStream<R>> {
        let initial_read = read_initial_bytes(&self.read_at, self.size().await).await?;
        let footer = initial_read.fb_footer()?;

        let row_count = footer.row_count()?;
        let footer_dtype = Arc::new(footer.lazily_projected_dtype(Projection::All)?);
        let read_projection = self.projection.unwrap_or_default();

        let projected_dtype = match read_projection {
            Projection::All => footer.dtype()?,
            Projection::Flat(ref projection) => footer.projected_dtype(projection)?,
        };

        let message_cache = Arc::new(RwLock::new(LayoutMessageCache::default()));

        let layout_reader = footer.layout_reader(
            Scan::new(match read_projection {
                Projection::All => None,
                Projection::Flat(p) => Some(Arc::new(Select::include(p))),
            }),
            RelativeLayoutCache::new(message_cache.clone(), footer_dtype.clone()),
        )?;

        let filter_reader = self
            .row_filter
            .map(|row_filter| {
                footer.layout_reader(
                    Scan::new(Some(Arc::new(row_filter))),
                    RelativeLayoutCache::new(message_cache.clone(), footer_dtype),
                )
            })
            .transpose()?;

        let row_mask = self
            .row_mask
            .as_ref()
            .map(|row_mask| {
                if row_mask.dtype().is_int() {
                    RowMask::from_index_array(row_mask, 0, row_count as usize)
                } else {
                    RowMask::from_mask_array(row_mask, 0, row_count as usize)
                }
            })
            .transpose()?;

        Ok(LayoutBatchStream::new(
            self.read_at,
            layout_reader,
            filter_reader,
            message_cache,
            projected_dtype,
            row_count,
            row_mask,
        ))
    }

    async fn size(&self) -> u64 {
        match self.size {
            Some(s) => s,
            None => self.read_at.size().await,
        }
    }
}
