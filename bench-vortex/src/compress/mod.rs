pub mod bench;
pub mod parquet;
pub mod vortex;

use std::sync::Arc;

use ::vortex::arrays::ChunkedArray;
use arrow_array::RecordBatch;
use arrow_schema::Schema;

pub fn chunked_to_vec_record_batch(chunked: ChunkedArray) -> (Vec<RecordBatch>, Arc<Schema>) {
    let chunks_vec = chunked.chunks();
    if chunks_vec.is_empty() {
        panic!("empty chunks");
    }
    let batches = chunks_vec
        .iter()
        .map(|x| RecordBatch::try_from(x.as_ref()).unwrap())
        .collect::<Vec<_>>();
    let schema = batches[0].schema();
    (batches, schema)
}
