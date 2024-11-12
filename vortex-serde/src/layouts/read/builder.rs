use std::sync::{Arc, RwLock};

use vortex_array::{Array, ArrayDType};
use vortex_error::VortexResult;
use vortex_expr::Select;
use vortex_schema::projection::Projection;

use super::RowMask;
use crate::io::VortexReadAt;
use crate::layouts::read::cache::{LayoutMessageCache, LazilyDeserializedDType, RelativeLayoutCache};
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
    row_mask: Option<Array>,
    row_filter: Option<RowFilter>,
}

impl<R: VortexReadAt> LayoutBatchStreamBuilder<R> {
    pub fn new(reader: R, layout_serde: LayoutDeserializer) -> Self {
        Self {
            reader,
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
        let footer = LayoutDescriptorReader::new(self.layout_serde.clone())
            .read_footer(&self.reader, self.size().await)
            .await?;
        let footer_dtype = Arc::new(LazilyDeserializedDType::from_schema_bytes(
            footer.dtype_bytes()?,
            Projection::All,
        ));
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
        let row_count = layout_reader.as_ref().row_count()?;

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
            self.reader,
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
            None => self.reader.size().await,
        }
    }
}
