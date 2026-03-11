use std::io::Cursor;

use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::arrow::FromArrowArray;
use vortex_array::ArrayRef;

use super::Fixture;

/// First partition of ClickBench hits, limited to 1000 rows.
const CLICKBENCH_URL: &str =
    "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_0.parquet";

pub struct ClickBenchHits1kFixture;

impl Fixture for ClickBenchHits1kFixture {
    fn name(&self) -> &str {
        "clickbench_hits_1k.vortex"
    }

    fn build(&self) -> Vec<ArrayRef> {
        let bytes = reqwest::blocking::get(CLICKBENCH_URL)
            .expect("failed to download ClickBench parquet")
            .bytes()
            .expect("failed to read ClickBench response body");

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("failed to open parquet")
            .with_batch_size(1000)
            .with_limit(1000)
            .build()
            .expect("failed to build parquet reader");

        let batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .expect("failed to read parquet batches");

        batches
            .into_iter()
            .map(|batch| ArrayRef::from_arrow(batch, false).expect("arrow conversion failed"))
            .collect()
    }
}
