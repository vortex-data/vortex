use duckdb::arrow::array::ArrayRef as ArrowArrayRef;
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use vortex_array::arrays::{ChunkedArray, ChunkedEncoding, VarBinViewArray, VarBinViewEncoding};
use vortex_array::arrow::FromArrowArray;
use vortex_array::compute::to_arrow_preferred;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayRef, ArrayStatistics, IntoArray, ToCanonical};
use vortex_dict::{DictArray, DictEncoding};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_fsst::{FSSTArray, FSSTEncoding};
use vortex_runend::{RunEndArray, RunEndEncoding};

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
    to_arrow_preferred(&canonical_array)?.to_duckdb(chunk, cache)
}

fn try_to_duckdb(
    array: &ArrayRef,
    chunk: &mut dyn WritableVector,
    cache: &mut ConversionCache,
) -> VortexResult<Option<()>> {
    if let Some(constant) = array.as_constant() {
        let value = constant.try_to_duckdb_scalar()?;
        chunk.flat_vector().assign_to_constant(&value);
        Ok(Some(()))
    } else if array.is_encoding(ChunkedEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<ChunkedArray>()
            .vortex_expect("ChunkedArray checked")
            .to_duckdb(chunk, cache)
            .map(Some)
    } else if array.is_encoding(VarBinViewEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<VarBinViewArray>()
            .vortex_expect("VarBinViewArray id checked")
            .to_duckdb(chunk, cache)
            .map(Some)
    } else if array.is_encoding(FSSTEncoding.id()) {
        let arr = array
            .as_any()
            .downcast_ref::<FSSTArray>()
            .vortex_expect("FSSTArray id checked");
        arr.to_varbinview()?.to_duckdb(chunk, cache).map(Some)
    } else if array.is_encoding(DictEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<DictArray>()
            .vortex_expect("DictArray id checked")
            .to_duckdb(chunk, cache)
            .map(Some)
    } else if array.is_encoding(RunEndEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<RunEndArray>()
            .vortex_expect("RunEndArray id checked")
            .to_duckdb(chunk, cache)
            .map(Some)
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
        let len = sized_vector.len;
        let arrow_arr = flat_vector_to_arrow_array(&mut sized_vector.vector, len)
            .map_err(|e| vortex_err!("Failed to convert duckdb array to vortex: {}", e))?;
        Ok(ArrayRef::from_arrow(arrow_arr, sized_vector.nullable))
    }
}
