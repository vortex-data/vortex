use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use crate::layouts::struct_::writer::StructLayoutWriter;
use crate::strategies::{LayoutStrategy, LayoutWriter};

/// Struct-of-chunks is the default Vortex layout strategy.
///
/// This layout first splits data into struct columns, before applying chunking as per the
/// provided batches.
///
/// TODO(ngates): add configuration options to this struct to re-chunk the data within each
///   column by size.
pub struct StructOfChunks;

impl LayoutStrategy for StructOfChunks {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        match dtype {
            DType::Struct(struct_dtype, nullability) => {
                if nullability == &Nullability::Nullable {
                    vortex_bail!("Structs with nullable fields are not supported");
                }

                Ok(Box::new(StructLayoutWriter::new(
                    dtype.clone(),
                    struct_dtype
                        .dtypes()
                        .map(|col_dtype| default_column_layout(&col_dtype))
                        .collect(),
                )))
            }
            _ => Ok(default_column_layout(dtype)),
        }
    }
}

fn default_column_layout(dtype: &DType) -> Box<dyn LayoutWriter> {
    Box::new(ChunkedLayoutWriter::new(
        dtype,
        ChunkedLayoutOptions::default(),
    )) as _
}
