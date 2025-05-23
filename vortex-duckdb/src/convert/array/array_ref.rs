use duckdb::arrow::array::ArrayRef as ArrowArrayRef;
use duckdb::core::LogicalTypeId;
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use vortex_array::arrays::{ChunkedVTable, DecimalArray, VarBinViewVTable};
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::compute::Cost;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_dict::DictVTable;
use vortex_error::{VortexResult, vortex_err};
use vortex_fsst::FSSTVTable;
use vortex_runend::RunEndVTable;

use crate::convert::array::cache::ConversionCache;
use crate::convert::array::data_chunk_adaptor::SizedFlatVector;
use crate::convert::scalar::ToDuckDBScalar;
use crate::{FromDuckDB, ToDuckDB};

pub fn to_duckdb(
    array: &ArrayRef,
    chunk: &mut dyn WritableVector,
    cache: &mut ConversionCache,
) -> VortexResult<()> {
    if try_to_duckdb(array, chunk, cache)?.is_some() {
        return Ok(());
    };
    let canonical_array = array.to_canonical()?.into_array();
    if try_to_duckdb(&canonical_array, chunk, cache)?.is_some() {
        return Ok(());
    };
    canonical_array
        .into_arrow_preferred()?
        .to_duckdb(chunk, cache)
}

fn try_to_duckdb(
    array: &ArrayRef,
    chunk: &mut dyn WritableVector,
    cache: &mut ConversionCache,
) -> VortexResult<Option<()>> {
    if array.is_constant_opts(Cost::Negligible) {
        let constant = array.scalar_at(0)?;
        let value = constant.try_to_duckdb_scalar()?;
        chunk.flat_vector().assign_to_constant(&value);
        Ok(Some(()))
    } else if array.dtype().is_decimal() {
        let decimal = array.to_decimal()?;
        decimal.to_duckdb(chunk, cache).map(Some)
    } else if let Some(array) = array.as_opt::<ChunkedVTable>() {
        array.to_duckdb(chunk, cache).map(Some)
    } else if let Some(array) = array.as_opt::<VarBinViewVTable>() {
        array.to_duckdb(chunk, cache).map(Some)
    } else if let Some(array) = array.as_opt::<FSSTVTable>() {
        array.to_varbinview()?.to_duckdb(chunk, cache).map(Some)
    } else if let Some(array) = array.as_opt::<DictVTable>() {
        array.to_duckdb(chunk, cache).map(Some)
    } else if let Some(array) = array.as_opt::<RunEndVTable>() {
        array.to_duckdb(chunk, cache).map(Some)
    } else {
        Ok(None)
    }
}

impl ToDuckDB for ArrowArrayRef {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        _: &mut ConversionCache,
    ) -> VortexResult<()> {
        write_arrow_array_to_vector(self, chunk)
            .map_err(|e| vortex_err!("Failed to convert vortex duckdb array: {}", e.to_string()))
    }
}

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
