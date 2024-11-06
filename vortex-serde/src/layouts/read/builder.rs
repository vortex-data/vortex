use std::sync::{Arc, RwLock};

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{Array, ArrayDType};
use vortex_error::VortexResult;
use vortex_expr::{Select, VortexExpr as _};
use vortex_schema::projection::Projection;

use super::RowMask;
use crate::io::VortexReadAt;
use crate::layouts::read::cache::{LayoutMessageCache, LazyDeserializedDType, RelativeLayoutCache};
use crate::layouts::read::context::LayoutDeserializer;
use crate::layouts::read::filtering::RowFilter;
use crate::layouts::read::footer::LayoutDescriptorReader;
use crate::layouts::read::stream::LayoutBatchStream;
use crate::layouts::read::Scan;

pub struct LayoutBatchStreamBuilder<R> {
    reader: R,
    layout_serde: LayoutDeserializer,
    projection: Option<Projection>,
    size: Option<u64>,
    indices: Option<Array>,
    row_filter: Option<RowFilter>,
}

impl<R: VortexReadAt> LayoutBatchStreamBuilder<R> {
    pub fn new(reader: R, layout_serde: LayoutDeserializer) -> Self {
        Self {
            reader,
            layout_serde,
            projection: None,
            row_filter: None,
            size: None,
            indices: None,
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

    pub async fn build(self) -> VortexResult<LayoutBatchStream<R>> {
        let footer = LayoutDescriptorReader::new(self.layout_serde.clone())
            .read_footer(&self.reader, self.size().await)
            .await?;
        let row_count = footer.row_count()?;
        let footer_dtype = Arc::new(LazyDeserializedDType::from_bytes(
            footer.dtype_bytes()?,
            Projection::All,
        ));
        let read_projection = self.projection.unwrap_or_default();

        let projected_dtype = match read_projection {
            Projection::All => footer.dtype()?,
            Projection::Flat(ref projection) => footer.projected_dtype(projection)?,
        };

        let message_cache = Arc::new(RwLock::new(LayoutMessageCache::default()));

        let data_reader = footer.layout(
            Scan::new(match read_projection {
                Projection::All => None,
                Projection::Flat(p) => Some(Arc::new(Select::include(p))),
            }),
            RelativeLayoutCache::new(message_cache.clone(), footer_dtype.clone()),
        )?;

        let row_filter_and_reader = match self.row_filter {
            None => None,
            Some(row_filter) => {
                let mut references = HashSet::new();
                row_filter.collect_references(&mut references);
                let select_filtering_columns =
                    Select::Include(references.into_iter().map(|x| x.clone()).collect());
                let layout = footer.layout(
                    Scan::new(Some(Arc::new(select_filtering_columns))),
                    RelativeLayoutCache::new(message_cache.clone(), footer_dtype),
                )?;
                Some((row_filter, layout))
            }
        };

        let indices_mask = self
            .indices
            .as_ref()
            .map(|indices| RowMask::from_index_array(indices, 0, row_count as usize))
            .transpose()?;

        Ok(LayoutBatchStream::new(
            self.reader,
            indices_mask,
            data_reader,
            row_filter_and_reader,
            message_cache,
            projected_dtype,
            row_count,
        ))
    }

    async fn size(&self) -> u64 {
        match self.size {
            Some(s) => s,
            None => self.reader.size().await,
        }
    }
}
