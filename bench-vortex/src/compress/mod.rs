// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use ::vortex::array::arrays::ChunkedArray;
use ::vortex::array::arrays::recursive_list_from_list_view;
use arrow_array::RecordBatch;
use arrow_schema::Schema;

pub mod bench;

pub mod parquet;
pub mod vortex;

#[cfg(feature = "lance")]
pub mod lance;

pub fn chunked_to_vec_record_batch(chunked: ChunkedArray) -> (Vec<RecordBatch>, Arc<Schema>) {
    let chunks_vec = chunked.chunks();
    assert!(!chunks_vec.is_empty(), "empty chunks");

    let batches = chunks_vec
        .iter()
        .map(|array| {
            // TODO(connor)[ListView]: The rust Parquet implementation does not support writing
            // `ListView` to Parquet files yet.
            let converted_array = recursive_list_from_list_view(array.clone());
            RecordBatch::try_from(converted_array.as_ref()).unwrap()
        })
        .collect::<Vec<_>>();

    let schema = batches[0].schema();
    (batches, schema)
}
