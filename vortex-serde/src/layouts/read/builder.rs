use std::io;
use std::sync::{Arc, RwLock};

use vortex::{Array, ArrayDType};
use vortex_error::VortexResult;
use vortex_schema::projection::Projection;

use crate::io::VortexReadAt;
use crate::layouts::read::cache::{LayoutMessageCache, LazyDeserializedDType, RelativeLayoutCache};
use crate::layouts::read::context::LayoutDeserializer;
use crate::layouts::read::filtering::RowFilter;
use crate::layouts::read::footer::LayoutDescriptorReader;
use crate::layouts::read::stream::LayoutBatchStream;
use crate::layouts::read::{Scan, DEFAULT_BATCH_SIZE};

pub struct LayoutReaderBuilder<R> {
    reader: R,
    layout_serde: LayoutDeserializer,
    projection: Option<Projection>,
    size: Option<u64>,
    indices: Option<Array>,
    row_filter: Option<RowFilter>,
    batch_size: Option<usize>,
}

impl<R: VortexReadAt> LayoutReaderBuilder<R> {
    pub fn new(reader: R, layout_serde: LayoutDeserializer) -> Self {
        Self {
            reader,
            layout_serde,
            projection: None,
            row_filter: None,
            size: None,
            indices: None,
            batch_size: None,
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
        // TODO(#441): Allow providing boolean masks
        assert!(
            array.dtype().is_int(),
            "Mask arrays have to be integer arrays"
        );
        self.indices = Some(array);
        self
    }

    pub fn with_row_filter(mut self, row_filter: RowFilter) -> Self {
        self.row_filter = Some(row_filter);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    // Build the stream, execute it, etc.
    pub async fn build(self) -> VortexResult<LayoutBatchStream<R>> {
        let footer = LayoutDescriptorReader::new(self.layout_serde.clone())
            .read_footer(&self.reader, self.size().await?)
            .await?;
        let batch_size = self.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        // TODO(robert): Propagate projection immediately instead of delegating to layouts, needs more restructuring
        let footer_dtype = Arc::new(LazyDeserializedDType::from_bytes(
            footer.dtype_bytes()?,
            Projection::All,
        ));
        let read_projection = self.projection.unwrap_or_default();

        let filter_projection = self
            .row_filter
            .as_ref()
            .map(|f| f.references().into_iter().cloned().collect::<Vec<_>>())
            .map(Projection::from);

        let projected_dtype = match read_projection {
            Projection::All => footer.dtype()?,
            Projection::Flat(ref projection) => footer.projected_dtype(projection)?,
        };

        let scan = Scan {
            filter: self.row_filter.clone(),
            batch_size,
            projection: read_projection,
            indices: self.indices,
        };

        let message_cache = Arc::new(RwLock::new(LayoutMessageCache::default()));

        let data_reader = footer.layout(
            scan.clone(),
            RelativeLayoutCache::new(message_cache.clone(), footer_dtype.clone()),
        )?;

        let filter_reader = filter_projection
            .map(|projection| {
                footer.layout(
                    Scan {
                        filter: self.row_filter,
                        batch_size,
                        projection,
                        indices: None,
                    },
                    RelativeLayoutCache::new(message_cache.clone(), footer_dtype),
                )
            })
            .transpose()?;

        Ok(LayoutBatchStream::new(
            self.reader,
            data_reader,
            filter_reader,
            message_cache,
            projected_dtype,
            scan,
        ))
    }

    async fn size(&self) -> io::Result<u64> {
        match self.size {
            Some(s) => Ok(s),
            None => self.reader.size().await,
        }
    }
}
