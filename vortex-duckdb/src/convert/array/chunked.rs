use duckdb::vtab::arrow::WritableVector;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, IntoArray};
use vortex_dict::DictArray;
use vortex_error::VortexResult;

use crate::ToDuckDB;
use crate::convert::array::array_ref::to_duckdb;
use crate::convert::array::cache::ConversionCache;

impl ToDuckDB for ChunkedArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        let chunks = self.chunks();
        if chunks.len() == 1 {
            return to_duckdb(&self.chunks()[0], chunk, cache);
        } else if chunks.len() == 2 {
            // It is common that a 2k split can contain a chunked array containing a pair of dictionaries
            // We this is special cased and handled without a usual canonical, once there is a way to
            // pre-canonicalize or cache, then this can be removed
            if let (Some(dict1), Some(dict2)) = (
                chunks[0].as_any().downcast_ref::<DictArray>(),
                chunks[1].as_any().downcast_ref::<DictArray>(),
            ) {
                let canon_values1 = cache.cached_array(dict1.values())?;
                let canon_values2 = cache.cached_array(dict2.values())?;

                let chunked = ChunkedArray::try_new(
                    vec![
                        DictArray::try_new(dict1.codes().clone(), canon_values1)?.into_array(),
                        DictArray::try_new(dict2.codes().clone(), canon_values2)?.into_array(),
                    ],
                    self.dtype().clone(),
                )?;
                return to_duckdb(&chunked.to_canonical()?.into_array(), chunk, cache);
            }
        }
        to_duckdb(&self.to_canonical()?.into_array(), chunk, cache)
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex_array::IntoArray;
    use vortex_array::arrays::{ChunkedArray, StructArray};
    use vortex_buffer::buffer;
    use vortex_dict::DictArray;

    use crate::{ConversionCache, to_duckdb_chunk};

    #[test]
    fn chunked_of_dict_to_duckdb() {
        let dict1 = DictArray::try_new(
            buffer![0u32, 1, 2, 2].into_array(),
            buffer![0, 1, 2, 3, 4, 5, 6, 7].into_array(),
        )
        .unwrap();
        let dict2 = DictArray::try_new(
            buffer![0u32, 1, 2, 2].into_array(),
            buffer![0, 1, 2, 3, 4, 5, 6, 7].into_array(),
        )
        .unwrap();
        let dtype = dict1.dtype().clone();
        let chunk =
            ChunkedArray::try_new(vec![dict1.into_array(), dict2.into_array()], dtype).unwrap();

        let sliced = chunk.slice(2, 7).unwrap();

        let struct_ = StructArray::from_fields(&[("a", sliced)]).unwrap();
        let mut cache = ConversionCache::default();
        let mut data_chunk =
            DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);

        to_duckdb_chunk(&struct_, &mut data_chunk, &mut cache).unwrap();

        assert_eq!(
            format!("{:?}", data_chunk),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 5 = [ 2, 2, 0, 1, 2]
"#
        )
    }
}
