// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use ::vortex::array::VortexSessionExecute;
use ::vortex::array::arrays::ChunkedArray;
use ::vortex::array::arrays::chunked::ChunkedArrayExt;
use ::vortex::array::arrays::listview::recursive_list_from_list_view;
use ::vortex::array::arrow::ArrowArrayExecutor;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
#[cfg(feature = "lance")]
pub use lance_bench::compress::LanceCompressor;
use vortex_bench::SESSION;

pub mod parquet;
pub mod vortex;

pub fn chunked_to_vec_record_batch(
    chunked: ChunkedArray,
) -> anyhow::Result<(Vec<RecordBatch>, Arc<Schema>)> {
    assert!(chunked.nchunks() > 0, "empty chunks");

    let batches = chunked
        .iter_chunks()
        .map(|array| {
            // TODO(connor)[ListView]: The rust Parquet implementation does not support writing
            // `ListView` to Parquet files yet.
            let converted_array = recursive_list_from_list_view(array.clone())?;
            let schema = converted_array.dtype().to_arrow_schema()?;
            Ok(converted_array
                .execute_record_batch(&schema, &mut SESSION.create_execution_ctx())?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let schema = batches[0].schema();
    Ok((batches, schema))
}
