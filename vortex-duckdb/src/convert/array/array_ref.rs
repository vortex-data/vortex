use duckdb::core::LogicalTypeId;
use duckdb::vtab::arrow::flat_vector_to_arrow_array;
use vortex::ArrayRef;
use vortex::arrays::DecimalArray;
use vortex::arrow::FromArrowArray;
use vortex::error::{VortexResult, vortex_err};

use crate::FromDuckDB;
use crate::convert::array::data_chunk_adaptor::SizedFlatVector;

impl FromDuckDB<SizedFlatVector> for ArrayRef {
    // TODO(joe): going via is slow, make it faster.
    fn from_duckdb(mut sized_vector: SizedFlatVector) -> VortexResult<ArrayRef> {
        if sized_vector.vector.logical_type().id() == LogicalTypeId::Decimal {
            return DecimalArray::from_duckdb(sized_vector);
        }

        let len = sized_vector.len;
        let arrow_arr = flat_vector_to_arrow_array(&mut sized_vector.vector, len)
            .map_err(|e| vortex_err!("Failed to convert duckdb array to vortex: {}", e))?;
        Ok(ArrayRef::from_arrow(
            arrow_arr.as_ref(),
            sized_vector.nullable,
        ))
    }
}
