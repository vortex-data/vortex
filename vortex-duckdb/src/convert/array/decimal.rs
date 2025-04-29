use duckdb::vtab::arrow::WritableVector;
use vortex_array::arrays::DecimalArray;
use vortex_error::VortexResult;

use crate::{ConversionCache, ToDuckDB};

impl ToDuckDB for DecimalArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        todo!("decimal")
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, FlatVector};
    use vortex_array::Array;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::convert::array::data_chunk_adaptor::SizedFlatVector;
    use crate::{to_duckdb_chunk, ConversionCache, ToDuckDB, ToDuckDBType};

    #[test]
    fn to_decimal() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );
        let chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        println!("vector {:?}", vector);
        to_duckdb_chunk()
        array
            .to_duckdb(
                &mut SizedFlatVector {
                    vector,
                    nullable: false,
                    len: array.len(),
                },
                &mut ConversionCache::default(),
            )
            .unwrap()
    }
}
