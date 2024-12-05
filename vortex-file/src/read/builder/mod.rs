use std::sync::{Arc, RwLock};

use initial_read::read_initial_bytes;
use vortex_array::{ArrayDType, ArrayData};
use vortex_error::VortexResult;
use vortex_expr::Select;
use vortex_io::{IoDispatcher, VortexReadAt};

use crate::read::cache::{LayoutMessageCache, RelativeLayoutCache};
use crate::read::context::LayoutDeserializer;
use crate::read::filtering::RowFilter;
use crate::read::projection::Projection;
use crate::read::stream::VortexFileArrayStream;
use crate::read::{RowMask, Scan};

pub(crate) mod initial_read;

/// Builder for reading Vortex files.
///
/// Succinctly, the file format specification is as follows:
///
/// 1. Data is written first, in a form that is describable by a Layout (typically Array IPC Messages).
///     a. To allow for more efficient IO & pruning, our writer implementation first writes the "data" arrays,
///        and then writes the "metadata" arrays (i.e., per-column statistics)
/// 2. We write what is collectively referred to as the "Footer", which contains:
///     a. An optional Schema, which if present is a valid flatbuffer representing a message::Schema
///     b. The Layout, which is a valid footer::Layout flatbuffer, and describes the physical byte ranges & relationships amongst
///        the those byte ranges that we wrote in part 1.
///     c. The Postscript, which is a valid footer::Postscript flatbuffer, containing the absolute start offsets of the Schema & Layout
///        flatbuffers within the file.
///     d. The End-of-File marker, which is 8 bytes, and contains the u16 version, u16 postscript length, and 4 magic bytes.
///
///
/// # Reified File Format
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
/// │     Schema Flatbuffer      │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │     Layout Flatbuffer      │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │    Postscript Flatbuffer   │
/// │  (Schema & Layout Offsets) │
/// │                            │
/// ├────────────────────────────┤
/// │     8-byte End of File     │
/// │(Version, Postscript Length,│
/// │       Magic Bytes)         │
/// └────────────────────────────┘
/// ```
pub struct VortexReadBuilder<R> {
    read_at: R,
    layout_serde: LayoutDeserializer,
    projection: Projection,
    file_size: Option<u64>,
    row_mask: Option<ArrayData>,
    row_filter: Option<RowFilter>,
    io_dispatcher: Option<Arc<IoDispatcher>>,
}

impl<R: VortexReadAt + Unpin> VortexReadBuilder<R> {
    pub fn new(read_at: R, layout_serde: LayoutDeserializer) -> Self {
        Self {
            read_at,
            layout_serde,
            projection: Projection::default(),
            file_size: None,
            row_mask: None,
            row_filter: None,
            io_dispatcher: None,
        }
    }

    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = Some(size);
        self
    }

    pub fn with_projection(mut self, projection: Projection) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_indices(mut self, array: ArrayData) -> Self {
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

    pub fn with_io_dispatcher(mut self, dispatcher: Arc<IoDispatcher>) -> Self {
        self.io_dispatcher = Some(dispatcher);
        self
    }

    pub async fn build(self) -> VortexResult<VortexFileArrayStream<R>> {
        // we do a large enough initial read to get footer, layout, and schema
        let initial_read = read_initial_bytes(&self.read_at, self.file_size().await?).await?;

        let layout = initial_read.fb_layout();

        let row_count = layout.row_count();
        let lazy_dtype = Arc::new(initial_read.lazy_dtype());

        let projected_dtype = match self.projection {
            Projection::All => lazy_dtype.clone(),
            Projection::Flat(ref fields) => lazy_dtype.project(fields)?,
        };

        let message_cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let layout_reader = self.layout_serde.read_layout(
            initial_read.fb_layout(),
            Scan::new(match self.projection {
                Projection::All => None,
                Projection::Flat(p) => Some(Arc::new(Select::include(p))),
            }),
            RelativeLayoutCache::new(message_cache.clone(), lazy_dtype.clone()),
        )?;

        let filter_reader = self
            .row_filter
            .map(|row_filter| {
                self.layout_serde.read_layout(
                    initial_read.fb_layout(),
                    Scan::new(Some(Arc::new(row_filter))),
                    RelativeLayoutCache::new(message_cache.clone(), lazy_dtype),
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

        // Default: fallback to single-threaded tokio dispatcher.
        let io_dispatcher = self.io_dispatcher.unwrap_or_default();

        VortexFileArrayStream::try_new(
            self.read_at,
            layout_reader,
            filter_reader,
            message_cache,
            projected_dtype,
            row_count,
            row_mask,
            io_dispatcher,
        )
    }

    async fn file_size(&self) -> VortexResult<u64> {
        Ok(match self.file_size {
            Some(s) => s,
            None => self.read_at.size().await?,
        })
    }
}
